// warning.js - PhishingNet

document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const blockedUrl = decodeURIComponent(urlParams.get('url') || 'Unknown URL');
    const confidence = parseFloat(urlParams.get('confidence')) || 0;
    const details = JSON.parse(decodeURIComponent(urlParams.get('details') || '[]'));

    document.getElementById('blockedUrl').textContent = blockedUrl;

    const confidencePercent = Math.round(confidence * 100);
    document.getElementById('confidenceText').textContent = confidencePercent + '%';

    const confidenceFill = document.getElementById('confidenceFill');
    confidenceFill.style.width = confidencePercent + '%';
    confidenceFill.classList.remove('phishing', 'suspicious', 'legitimate');

    if (confidence >= 0.65) {
        confidenceFill.classList.add('phishing');
    } else if (confidence >= 0.45) {
        confidenceFill.classList.add('suspicious');
    } else {
        confidenceFill.classList.add('legitimate');
    }

    const statusBadge = document.getElementById('statusBadge');
    const tierBadge = document.getElementById('tierBadge');

    if (confidence >= 0.65) {
        statusBadge.textContent = '🚨 Phishing Detected!';
        statusBadge.className = 'subtitle phishing';
        tierBadge.textContent = '🚨 PHISHING';
        tierBadge.className = 'status-badge phishing';
    } else if (confidence >= 0.45) {
        statusBadge.textContent = '⚠️ Suspicious URL';
        statusBadge.className = 'subtitle suspicious';
        tierBadge.textContent = '⚠️ SUSPICIOUS';
        tierBadge.className = 'status-badge suspicious';
    } else {
        statusBadge.textContent = '✅ Legitimate URL';
        statusBadge.className = 'subtitle legitimate';
        tierBadge.textContent = '✅ LEGITIMATE';
        tierBadge.className = 'status-badge legitimate';
    }

    const reasonList = document.getElementById('reasonList');
    reasonList.innerHTML = '';
    const reasons = details.length ? details : ['Suspicious URL pattern detected'];
    reasons.forEach(reason => {
        const li = document.createElement('li');
        li.textContent = reason;
        reasonList.appendChild(li);
    });

    window._blockedUrl = blockedUrl;
});

// ===================== BUTTON FUNCTIONS =====================
function goBack() {
    if (window.history.length > 1) {
        window.history.back();
    } else {
        window.location.href = 'https://www.google.com';
    }
}

function continueAnyway() {
    const url = window._blockedUrl;
    if (url) {
        // Send message to background script to whitelist this URL briefly
        chrome.runtime.sendMessage({ 
            action: 'continueAnyway', 
            url: url 
        }, function(response) {
            // Redirect to the URL after the message is sent
            window.location.href = url;
        });
    } else {
        window.history.back();
    }
}

// ===================== ATTACH EVENT LISTENERS =====================
setTimeout(function() {
    const goBackBtn = document.getElementById('goBackBtn');
    const continueBtn = document.getElementById('continueBtn');
    if (goBackBtn) goBackBtn.addEventListener('click', goBack);
    if (continueBtn) continueBtn.addEventListener('click', continueAnyway);
}, 100);