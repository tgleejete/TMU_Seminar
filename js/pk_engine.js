/**
 * pk_engine.js  v2 — TMU PopPK Lab
 *
 * FIX: replaced while-loop (t accumulation) with for-loop on integer grid index.
 * Root cause of Module 01/02 freeze: t = +(t + hActual).toFixed(6) truncates
 * sub-microsecond remainders back to the same value → hActual = ~5e-8 forever.
 */

// ─── RK4 solver ──────────────────────────────────────────────────────────────
function rk4Step(dydt, y, t, h) {
  const k1 = dydt(t,        y);
  const k2 = dydt(t + h/2,  y.map((v,i) => v + h/2 * k1[i]));
  const k3 = dydt(t + h/2,  y.map((v,i) => v + h/2 * k2[i]));
  const k4 = dydt(t + h,    y.map((v,i) => v + h   * k3[i]));
  return y.map((v,i) => v + h/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));
}

/**
 * Simulate PK — multi-dose, IV bolus / infusion / oral, 1-CMT
 */
function simulate(events, params, tEnd=168, dt=0.1) {
  const { CL, V, ka=1.5 } = params;
  const ke = CL / V;

  // Use integer grid: t = i * dt — eliminates floating-point drift entirely
  const nSteps = Math.ceil(tEnd / dt);
  const evts = [...events].sort((a,b) => a.time - b.time);

  let y = [0, 0]; // [DEPOT, CENT]
  const result = [];
  let infusions = [];
  let evtIdx = 0;

  const dydt = (t, y) => {
    const inf = infusions.filter(r => t < r.endTime).reduce((s,r) => s + r.rate, 0);
    return [-ka * y[0],  ka * y[0] + inf/V - ke * y[1]];
  };

  for (let i = 0; i <= nSteps; i++) {
    const t = +(i * dt).toFixed(4); // exact, no accumulation

    while (evtIdx < evts.length && evts[evtIdx].time <= t + 1e-9) {
      const ev = evts[evtIdx++];
      if (ev.route === 'oral') {
        y[0] += ev.amt;
      } else if (ev.route === 'iv') {
        if (!ev.dur || ev.dur === 0) {
          y[1] += ev.amt / V;                                    // bolus
        } else {
          infusions.push({ endTime: ev.time + ev.dur, rate: ev.amt / ev.dur }); // mg/h
        }
      }
    }

    result.push({ t, c: Math.max(0, y[1]) });
    if (i < nSteps) y = rk4Step(dydt, y, t, dt);
  }
  return result;
}

// ─── Covariate model ─────────────────────────────────────────────────────────
function popCL(TVCL, WT=70, ALB=40, thetaALB=0) {
  return TVCL * Math.pow(WT/70, 0.75) * Math.exp(thetaALB * (ALB-40)/10);
}

// ─── MAP Bayesian OFV ────────────────────────────────────────────────────────
function computeOFV(eta, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events) {
  const CL = cl_pop * Math.exp(eta);
  const tMax = Math.max(...observations.map(o=>o.t)) + 1;
  const sim = simulate(events, { CL, V }, tMax);
  let residSum = 0;
  for (const obs of observations) {
    const row = sim.reduce((a,b) => Math.abs(b.t-obs.t) < Math.abs(a.t-obs.t) ? b : a);
    const ipred = row ? row.c : 0;
    const sigSq = sigma_prop2 * ipred * ipred + sigma_add2;
    if (sigSq > 0) residSum += Math.pow(obs.dv - ipred, 2) / sigSq;
  }
  return (eta * eta) / omega2 + residSum;
}

function mapEstimate(cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events) {
  let best = 0, bestOFV = Infinity;
  for (let e = -3; e <= 3; e += 0.01) {
    const v = computeOFV(e, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events);
    if (v < bestOFV) { bestOFV = v; best = e; }
  }
  for (let e = best-0.1; e <= best+0.1; e += 0.001) {
    const v = computeOFV(e, cl_pop, V, omega2, sigma_prop2, sigma_add2, observations, events);
    if (v < bestOFV) { bestOFV = v; best = e; }
  }
  return { eta: +best.toFixed(4), ofv: +bestOFV.toFixed(4) };
}

// ─── MIPD dose optimiser ─────────────────────────────────────────────────────
function findOptimalDose(targetC, tau, params, route='iv', nDoses=6) {
  let lo=1, hi=2000;
  for (let iter=0; iter<40; iter++) {
    const mid = (lo+hi)/2;
    const events = Array.from({length:nDoses},(_,i)=>({time:i*tau, amt:mid, route, dur:0}));
    const sim = simulate(events, params, nDoses*tau+1);
    const tTrough = (nDoses-1)*tau + tau - 0.1;
    const row = sim.reduce((a,b) => Math.abs(b.t-tTrough)<Math.abs(a.t-tTrough) ? b : a);
    if (row.c < targetC) lo = mid; else hi = mid;
  }
  return Math.round((lo+hi)/2);
}

if (typeof module !== 'undefined') module.exports = { simulate, popCL, computeOFV, mapEstimate, findOptimalDose };
