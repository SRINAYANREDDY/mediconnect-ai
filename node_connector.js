/**
 * MediConnect AI Connector
 * Drop this file into mediconnect-backend/src/utils/aiConnector.js
 * Node.js backend uses this to call the Python AI microservice.
 */

const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const AI_BASE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';
const AI_API_KEY  = process.env.AI_API_KEY || 'mediconnect_ai_secret_key_change_in_production';

const aiClient = axios.create({
  baseURL: AI_BASE_URL,
  timeout: 30000,
  headers: { 'X-AI-API-Key': AI_API_KEY },
});

/**
 * Analyse blood test parameters
 * @param {Object} parameters - e.g. { haemoglobin: 9.2, wbc: 11200, ... }
 * @param {string} reportType - "CBC" | "LFT" | "KFT" | "Thyroid" | "Diabetes" | "Lipid"
 * @param {number} patientAge
 * @param {string} patientGender - "M" | "F"
 */
async function analyseBlood(parameters, reportType = 'CBC', patientAge = null, patientGender = null) {
  try {
    const res = await aiClient.post('/analyse/blood', {
      parameters,
      report_type: reportType,
      patient_age: patientAge,
      patient_gender: patientGender,
    });
    return { success: true, data: res.data };
  } catch (err) {
    console.error('AI blood analysis error:', err.message);
    return { success: false, error: err.message };
  }
}

/**
 * Analyse X-Ray / CT / MRI image file
 * @param {string} filePath - path to uploaded image file
 * @param {string} reportType - "Chest X-Ray" | "Bone X-Ray" | "CT Chest" | "MRI Brain"
 */
async function analyseXray(filePath, reportType = 'Chest X-Ray', patientAge = null, patientGender = null, clinicalNotes = '') {
  try {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    form.append('report_type', reportType);
    if (patientAge)     form.append('patient_age', String(patientAge));
    if (patientGender)  form.append('patient_gender', patientGender);
    if (clinicalNotes)  form.append('clinical_notes', clinicalNotes);

    const res = await aiClient.post('/analyse/xray', form, {
      headers: { ...form.getHeaders(), 'X-AI-API-Key': AI_API_KEY },
      timeout: 60000,
    });
    return { success: true, data: res.data };
  } catch (err) {
    console.error('AI X-ray analysis error:', err.message);
    return { success: false, error: err.message };
  }
}

/**
 * Analyse ECG parameters
 * @param {Object} ecgData - heart_rate, pr_interval, st_elevation, etc.
 */
async function analyseECG(ecgData) {
  try {
    const res = await aiClient.post('/analyse/ecg', ecgData);
    return { success: true, data: res.data };
  } catch (err) {
    console.error('AI ECG analysis error:', err.message);
    return { success: false, error: err.message };
  }
}

/**
 * Analyse text-based report (biopsy, psych assessment)
 * @param {string} reportType - "Biopsy" | "Psychological Assessment"
 * @param {string} reportText - full text content of the report
 */
async function analyseNLP(reportType, reportText, patientAge = null, patientGender = null) {
  try {
    const res = await aiClient.post('/analyse/nlp', {
      report_type: reportType,
      report_text: reportText,
      patient_age: patientAge,
      patient_gender: patientGender,
    });
    return { success: true, data: res.data };
  } catch (err) {
    console.error('AI NLP analysis error:', err.message);
    return { success: false, error: err.message };
  }
}

/** Check if AI service is running */
async function healthCheck() {
  try {
    const res = await aiClient.get('/health');
    return res.data;
  } catch (err) {
    return { status: 'unreachable', error: err.message };
  }
}

module.exports = { analyseBlood, analyseXray, analyseECG, analyseNLP, healthCheck };
