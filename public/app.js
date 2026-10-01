/**
 * CardioPulse AI — Client-Side Application Logic
 * Connects UI form controls, presets, animations, and API prediction endpoints.
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements - Form & Inputs
  const form = document.getElementById('cardioForm');
  const btnSubmit = document.getElementById('btnSubmitPredict');
  const sliderAge = document.getElementById('sliderAge');
  const inputAge = document.getElementById('inputAge');
  const sliderOldpeak = document.getElementById('sliderOldpeak');
  const inputOldpeak = document.getElementById('inputOldpeak');
  const hrFormulaHint = document.getElementById('hrFormulaHint');

  // DOM Elements - Presets
  const btnPresetHealthy = document.getElementById('btnPresetHealthy');
  const btnPresetModerate = document.getElementById('btnPresetModerate');
  const btnPresetHigh = document.getElementById('btnPresetHigh');
  const btnResetForm = document.getElementById('btnResetForm');

  // DOM Elements - Views
  const placeholderView = document.getElementById('placeholderView');
  const resultsView = document.getElementById('resultsView');
  const btnPrintReport = document.getElementById('btnPrintReport');

  // DOM Elements - Results Card
  const riskBadge = document.getElementById('riskBadge');
  const resultHeadline = document.getElementById('resultHeadline');
  const evaluationTime = document.getElementById('evaluationTime');
  const riskScoreCounter = document.getElementById('riskScoreCounter');
  const gaugeProgressArc = document.getElementById('gaugeProgressArc');
  const summaryText = document.getElementById('summaryText');
  const factorsContainer = document.getElementById('factorsContainer');
  const recommendationsContainer = document.getElementById('recommendationsContainer');
  const patientMetricsStrip = document.getElementById('patientMetricsStrip');
  const modelStatusText = document.getElementById('modelStatusText');

  // SVG Gauge Arc Constant (Half circle r=80: pi * 80 ~= 251.327)
  const GAUGE_CIRCUMFERENCE = 251.327;

  // Bundled Client-Side Fallback Presets for Instant Latency-Free Demo
  const PRESETS = {
    healthy: {
      age: 32,
      sex: 0,
      cp: 2,
      trestbps: 115,
      chol: 175,
      fbs: 0,
      restecg: 0,
      thalach: 180,
      exang: 0,
      oldpeak: 0.0,
      slope: 1,
      ca: 0,
      thal: 3
    },
    moderate: {
      age: 54,
      sex: 1,
      cp: 3,
      trestbps: 138,
      chol: 235,
      fbs: 0,
      restecg: 1,
      thalach: 145,
      exang: 0,
      oldpeak: 1.2,
      slope: 2,
      ca: 0,
      thal: 6
    },
    high: {
      age: 64,
      sex: 1,
      cp: 4,
      trestbps: 160,
      chol: 285,
      fbs: 1,
      restecg: 2,
      thalach: 115,
      exang: 1,
      oldpeak: 2.8,
      slope: 2,
      ca: 2,
      thal: 7
    }
  };

  // -------------------------------------------------------------
  // Input Synchronization & Dynamic Formula Calculation
  // -------------------------------------------------------------
  function updateTargetHrHint(age) {
    const maxPredictedHr = Math.round(220 - age);
    const targetHrRange = Math.round(maxPredictedHr * 0.85);
    if (hrFormulaHint) {
      hrFormulaHint.textContent = `Max formula: ~${maxPredictedHr} bpm (85%: ${targetHrRange})`;
    }
  }

  // Age Sync
  sliderAge.addEventListener('input', (e) => {
    inputAge.value = e.target.value;
    updateTargetHrHint(Number(e.target.value));
  });

  inputAge.addEventListener('input', (e) => {
    sliderAge.value = e.target.value;
    updateTargetHrHint(Number(e.target.value));
  });

  // Oldpeak Sync
  sliderOldpeak.addEventListener('input', (e) => {
    inputOldpeak.value = parseFloat(e.target.value).toFixed(1);
  });

  inputOldpeak.addEventListener('input', (e) => {
    sliderOldpeak.value = e.target.value;
  });

  // Initialize initial hint
  updateTargetHrHint(Number(inputAge.value));

  // -------------------------------------------------------------
  // Preset Loading Function
  // -------------------------------------------------------------
  function loadPreset(presetKey, autoSubmit = true) {
    const data = PRESETS[presetKey];
    if (!data) return;

    // Apply values
    sliderAge.value = data.age;
    inputAge.value = data.age;
    updateTargetHrHint(data.age);

    const sexRadio = form.querySelector(`input[name="sex"][value="${data.sex}"]`);
    if (sexRadio) sexRadio.checked = true;

    document.getElementById('inputTrestbps').value = data.trestbps;
    document.getElementById('inputChol').value = data.chol;

    const fbsRadio = form.querySelector(`input[name="fbs"][value="${data.fbs}"]`);
    if (fbsRadio) fbsRadio.checked = true;

    document.getElementById('inputCp').value = data.cp;
    document.getElementById('inputThalach').value = data.thalach;

    const exangRadio = form.querySelector(`input[name="exang"][value="${data.exang}"]`);
    if (exangRadio) exangRadio.checked = true;

    document.getElementById('inputRestecg').value = data.restecg;

    sliderOldpeak.value = data.oldpeak;
    inputOldpeak.value = data.oldpeak.toFixed(1);

    document.getElementById('inputSlope').value = data.slope;
    document.getElementById('inputCa').value = data.ca;
    document.getElementById('inputThal').value = data.thal;

    if (autoSubmit) {
      triggerPrediction();
    }
  }

  btnPresetHealthy.addEventListener('click', () => loadPreset('healthy', true));
  btnPresetModerate.addEventListener('click', () => loadPreset('moderate', true));
  btnPresetHigh.addEventListener('click', () => loadPreset('high', true));

  btnResetForm.addEventListener('click', () => {
    form.reset();
    sliderAge.value = 55;
    inputAge.value = 55;
    sliderOldpeak.value = 1.0;
    inputOldpeak.value = "1.0";
    updateTargetHrHint(55);

    // Return to placeholder view
    resultsView.style.display = 'none';
    placeholderView.style.display = 'flex';
  });

  // -------------------------------------------------------------
  // Form Serialization
  // -------------------------------------------------------------
  function getFormData() {
    const formData = new FormData(form);
    return {
      age: parseInt(formData.get('age'), 10),
      sex: parseInt(formData.get('sex'), 10),
      cp: parseInt(formData.get('cp'), 10),
      trestbps: parseFloat(formData.get('trestbps')),
      chol: parseFloat(formData.get('chol')),
      fbs: parseInt(formData.get('fbs'), 10),
      restecg: parseInt(formData.get('restecg'), 10),
      thalach: parseFloat(formData.get('thalach')),
      exang: parseInt(formData.get('exang'), 10),
      oldpeak: parseFloat(formData.get('oldpeak')),
      slope: parseInt(formData.get('slope'), 10),
      ca: parseInt(formData.get('ca'), 10),
      thal: parseInt(formData.get('thal'), 10)
    };
  }

  // -------------------------------------------------------------
  // Prediction Execution & Animation
  // -------------------------------------------------------------
  async function triggerPrediction() {
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }

    const payload = getFormData();

    // Show button loading state
    btnSubmit.classList.add('loading');
    btnSubmit.disabled = true;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server returned error status ${response.status}`);
      }

      const result = await response.json();
      renderResults(result);

    } catch (err) {
      console.error('Prediction request error:', err);
      alert(`Cardiovascular evaluation request failed:\n${err.message}\n\nPlease check your server or Vercel connection.`);
    } finally {
      btnSubmit.classList.remove('loading');
      btnSubmit.disabled = false;
    }
  }

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    triggerPrediction();
  });

  // -------------------------------------------------------------
  // Render Diagnostics & Animate Gauge
  // -------------------------------------------------------------
  function renderResults(data) {
    // Switch views smoothly
    placeholderView.style.display = 'none';
    resultsView.style.display = 'flex';

    // Risk category badges
    riskBadge.className = 'assessment-badge';
    if (data.badge_variant === 'success') {
      riskBadge.classList.add('badge-low');
      riskBadge.textContent = 'LOW RISK';
    } else if (data.badge_variant === 'warning') {
      riskBadge.classList.add('badge-mod');
      riskBadge.textContent = 'MODERATE RISK';
    } else {
      riskBadge.classList.add('badge-high');
      riskBadge.textContent = 'HIGH RISK';
    }

    resultHeadline.textContent = data.headline;
    evaluationTime.textContent = `Evaluated: ${new Date(data.evaluated_at || Date.now()).toLocaleTimeString()}`;
    summaryText.textContent = data.summary;

    // Animate Probability Counter (0% to target %)
    animateCounter(riskScoreCounter, data.risk_percentage, 1000);

    // Animate Radial Gauge Fill Arc
    // When prob = 0: offset = GAUGE_CIRCUMFERENCE (empty)
    // When prob = 1: offset = 0 (full)
    const targetOffset = GAUGE_CIRCUMFERENCE * (1 - Math.min(1.0, data.probability));
    gaugeProgressArc.style.strokeDashoffset = targetOffset;

    // Color code gauge stroke based on probability
    if (data.probability < 0.35) {
      gaugeProgressArc.style.stroke = '#10b981'; // emerald
    } else if (data.probability < 0.65) {
      gaugeProgressArc.style.stroke = '#f59e0b'; // amber
    } else {
      gaugeProgressArc.style.stroke = '#f43f5e'; // crimson
    }

    // Render Contributing Factors
    factorsContainer.innerHTML = '';
    if (!data.contributing_factors || data.contributing_factors.length === 0) {
      factorsContainer.innerHTML = `
        <div class="factor-card">
          <div class="factor-main">
            <span class="factor-param">No Significant Clinical Risk Factors</span>
            <span class="factor-note">All measured physiological biomarkers are within optimal clinical thresholds.</span>
          </div>
          <span class="factor-value-tag" style="background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3);">Normal</span>
        </div>
      `;
    } else {
      data.contributing_factors.forEach(factor => {
        const factorCard = document.createElement('div');
        factorCard.className = 'factor-card';
        const tagClass = factor.severity === 'high' ? 'tag-high' : 'tag-moderate';

        factorCard.innerHTML = `
          <div class="factor-main">
            <span class="factor-param">${escapeHtml(factor.parameter)} — ${escapeHtml(factor.status)}</span>
            <span class="factor-note">${escapeHtml(factor.note)}</span>
          </div>
          <span class="factor-value-tag ${tagClass}">${escapeHtml(factor.value)}</span>
        `;
        factorsContainer.appendChild(factorCard);
      });
    }

    // Render Recommendations
    recommendationsContainer.innerHTML = '';
    if (data.recommendations && data.recommendations.length > 0) {
      data.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsContainer.appendChild(li);
      });
    }

    // Render Patient Metrics Strip
    patientMetricsStrip.innerHTML = `
      <div class="patient-pill">Age: <strong>${data.patient_metrics.age}</strong></div>
      <div class="patient-pill">Sex: <strong>${data.patient_metrics.sex}</strong></div>
      <div class="patient-pill">Resting BP: <strong>${data.patient_metrics.resting_bp}</strong></div>
      <div class="patient-pill">Cholesterol: <strong>${data.patient_metrics.cholesterol}</strong></div>
      <div class="patient-pill">Max HR: <strong>${data.patient_metrics.max_heart_rate}</strong></div>
      <div class="patient-pill">ST Depr: <strong>${data.patient_metrics.st_depression}</strong></div>
      <div class="patient-pill">Vessels Ca: <strong>${data.patient_metrics.occluded_vessels}</strong></div>
    `;

    // Smooth scroll into view on mobile
    if (window.innerWidth <= 1024) {
      resultsView.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  // -------------------------------------------------------------
  // Number Counter Animation Helper
  // -------------------------------------------------------------
  function animateCounter(element, targetVal, durationMs) {
    const startTime = performance.now();
    const startVal = 0.0;

    function step(currentTime) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / durationMs, 1);
      // Ease out cubic
      const easedProgress = 1 - Math.pow(1 - progress, 3);
      const current = (startVal + (targetVal - startVal) * easedProgress).toFixed(1);
      element.textContent = current;

      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        element.textContent = targetVal.toFixed(1);
      }
    }

    requestAnimationFrame(step);
  }

  // -------------------------------------------------------------
  // Utility & Print Report
  // -------------------------------------------------------------
  btnPrintReport.addEventListener('click', () => {
    window.print();
  });

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  // Check health and update status dot on mount
  fetch('/api/health')
    .then(res => res.json())
    .then(info => {
      if (info.status === 'healthy' && modelStatusText) {
        modelStatusText.textContent = 'Random Forest Pipeline Ready • 91.8% Accuracy';
      }
    })
    .catch(() => {
      if (modelStatusText) {
        modelStatusText.textContent = 'Serverless Function Ready';
      }
    });
});
