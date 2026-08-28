// warning.js - PhishingNet (Updated for Three-Tier System)

document.addEventListener('DOMContentLoaded', function() {
    // Get parameters from URL
    const urlParams = new URLSearchParams(window.location.search);
    const blockedUrl = decodeURIComponent(urlParams.get('url') || 'Unknown URL');
    const confidence = parseFloat(urlParams.get('confidence')) || 0;
    const explanation = decodeURIComponent(urlParams.get('explanation') || '');
    const details = JSON.parse(decodeURIComponent(urlParams.get('details') || '[]'));

    // Display blocked URL
    document.getElementById('blockedUrl').textContent = blockedUrl;

    // Display confidence
    const confidencePercent = Math.round(confidence * 100);
    document.getElementById('confidenceText').textContent = confidencePercent + '%';
    document.getElementById('confidenceFill').style.width = confidencePercent + '%';

    // Display reasons
    const reasonList = document.getElementById('reasonList');
    reasonList.innerHTML = '';

    let reasons = [];
    if (details && details.length > 0) {
        reasons = details;
    } else if (explanation) {
        reasons = [explanation];
    } else {
        reasons = ['Suspicious URL pattern detected'];
    }

    reasons.forEach(reason => {
        const li = document.createElement('li');
        li.textContent = reason;
        reasonList.appendChild(li);
    });

    // Store blocked URL globally
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
        try {
            chrome.runtime.sendMessage({
                action: 'continueAnyway',
                url: url
            });
        } catch (e) {
            // Ignore - message sending not required
        }
        window.location.href = url;
    } else {
        window.history.back();
    }
}

// ===================== ATTACH BUTTON EVENTS =====================

setTimeout(function() {
    const goBackBtn = document.getElementById('goBackBtn');
    const continueBtn = document.getElementById('continueBtn');
    
    if (goBackBtn) {
        goBackBtn.addEventListener('click', goBack);
    }
    
    if (continueBtn) {
        continueBtn.addEventListener('click', continueAnyway);
    }
}, 100);