/**
 * pk_engine.js
 * Shared pharmacokinetic computation core for TMU PopPK Lab
 * Supports: multi-dose, IV bolus / IV infusion / oral, 1-compartment
 */

// ─── ODE solver (RK4, fixed step) ───────────────────────────────────────────
function rk4Step(dydt, y, t, h) {
  const k1 = dydt(t,        y);
  const k2 = dydt(t + h/2,  y.map((v,i) => v + h/2 * k1[i]));
  const k3 = dydt(t + h/2,  y.map((v,i) => v + h/2 * k2[i]));
  const k4 = dydt(t + h,    y.map((v,i) => v + h   * k3[i]));
  return y.map((v,i) => v + h/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
}

/**
 * Simulate PK profile with arbitrary dose history
 * @param {Array}  events  [{time, amt, route:'iv'|'oral', dur:0}]
 * @param {Object} params  {CL, V, ka}  (ka only for oral)
 * @param {number} tEnd
 * @param {number} dt      step size (h)
 * @returns {Array} [{t, c}]
 */
function simulate(events, params, tEnd=168, dt=0.1) {
  const { CL, V, ka=1.5 } = params;
  const ke = CL / V;

  // Sort events ascending
  const evts = [...events].sort((a,b) => a.time - b.time);

  // State: [DEPOT, CENT]  (DEPOT only used for oral)
  let y = [0, 0];
  const result = [];
  let t = 0;

  // Active infusions: [{endTime, rate}]
  let infusions = [];

  const step = (t, y) => {
    // Sum infusion inputs active at time t (rate in mg/h, divide by V → mg/L/h)
    const inf = infusions.filter(r => t < r.endTime).reduce((s,r) => s+r.rate, 0);
    const dDEPOT = -ka * y[0];
    const dCENT  = ka * y[0] + inf/V - ke * y[1];
    return [dDEPOT, dCENT];
  };

  let evtIdx = 0;

  while (t <= tEnd + 1e-9) {
    // Apply any events at this time
    while (evtIdx < evts.length && evts[evtIdx].time <= t + 1e-9) {
      const ev = evts[evtIdx++];
      if (ev.route === 'oral') {
        y[0] += ev.amt;
      } else if (ev.route === 'iv') {
        if (!ev.dur || ev.dur === 0) {
          // IV bolus
          y[1] += ev.amt / V;
        } else {
          // IV infusion: store rate as mg/h (divide by V happens in step())
          infusions.push({ endTime: ev.time + ev.dur, rate: ev.amt / ev.dur });
        }
      }
    }

    result.push({ t: +t.toFixed(4), c: Math.max(0, y[1]) });

    if (t >= tEnd) break;
    const hActual = Math.min(dt, tEnd - t);
    y = rk4Step(step, y, t, hActual);
    t = +(t + hActual).toFixed(6);
  }

  return result;
}

// ─── MAP Bayesian estimation ─────────────────────────────────────────────────

/**
 * Compute population CL with covariates
 * CL_pop = TVCL * (WT/70)^0.75 * exp(theta_ALB * (ALB-40)/10)
 */
function popCL(TVCL, WT=70, ALB=40, thetaALB=0) {
  return TVCL * Math.pow(WT/70, 0.75) * Math.exp(thetaALB * (ALB-40)/10);
}

/**
 * Objective function value for MAP (extended least squares)
 * OFV = η²/ω² + Σ (DV-IPRED)² / (σ_prop²·IPRED² + σ_add²)
 */
function computeOFV(eta, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events) {
  const CL = cl_pop * Math.exp(eta);
  const sim = simulate(events, { CL, V }, Math.max(...observations.map(o=>o.t)) + 1);

  let residSum = 0;
  for (const obs of observations) {
    const row = sim.find(r => Math.abs(r.t - obs.t) < 0.15);
    const ipred = row ? row.c : 0;
    const sigSq = sigma_prop2 * ipred * ipred + sigma_add2;
    if (sigSq > 0) residSum += Math.pow(obs.dv - ipred, 2) / sigSq;
  }

  return (eta * eta) / omega2 + residSum;
}

/**
 * Grid-search MAP estimator (golden section refinement)
 */
function mapEstimate(cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events) {
  let best = 0, bestOFV = Infinity;
  for (let e = -3; e <= 3; e += 0.01) {
    const v = computeOFV(e, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events);
    if (v < bestOFV) { bestOFV = v; best = e; }
  }
  // Refine around best ± 0.1 in steps of 0.001
  for (let e = best-0.1; e <= best+0.1; e += 0.001) {
    const v = computeOFV(e, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events);
    if (v < bestOFV) { bestOFV = v; best = e; }
  }
  return { eta: +best.toFixed(4), ofv: +bestOFV.toFixed(4) };
}

// ─── MIPD dose finder ────────────────────────────────────────────────────────

/**
 * Find dose that achieves target trough (Cmin at tau)
 * @param {number} targetC  target concentration (mg/L)
 * @param {number} tau      dosing interval (h)
 * @param {Object} params   {CL, V, ka, route}
 * @param {number} nDoses   number of doses to simulate (steady state)
 */
function findOptimalDose(targetC, tau, params, route='iv', nDoses=6) {
  let lo=1, hi=2000;
  for (let iter=0; iter<40; iter++) {
    const mid = (lo+hi)/2;
    const events = Array.from({length:nDoses},(_,i)=>({time:i*tau,amt:mid,route,dur:0}));
    const sim = simulate(events, params, nDoses*tau+1);
    // Cmin = concentration just before last dose
    const tTrough = (nDoses-1)*tau + tau - 0.05;
    const row = sim.reduce((a,b) => Math.abs(b.t-tTrough)<Math.abs(a.t-tTrough)?b:a);
    if (row.c < targetC) lo = mid; else hi = mid;
  }
  return Math.round((lo+hi)/2);
}

// Export for module use (ignored in browser)
if (typeof module !== 'undefined') module.exports = { simulate, popCL, computeOFV, mapEstimate, findOptimalDose };
