// background.js - PhishingNet Service Worker

const API_URL = 'https://phishing-detection-api-j1dv.onrender.com/predict';
const CACHE_EXPIRY = 3600000; // 1 hour in milliseconds

// ===================== WHITELIST (Local Pre-check) =====================

const WHITELIST = [
    // Google
    'google.com', 'google.co.in', 'gmail.com', 'youtube.com',
    'drive.google.com', 'docs.google.com',
    
    // Social Media
    'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com',
    'reddit.com', 'github.com', 'tiktok.com', 'whatsapp.com',
    'telegram.org', 'discord.com', 'twitch.tv',
    
    // E-commerce
    'amazon.com', 'amazon.in', 'flipkart.com', 'walmart.com',
    'target.com', 'shopify.com', 'etsy.com', 'ebay.com',
    'alibaba.com', 'aliexpress.com',
    
    // Banking
    'paypal.com', 'hdfcbank.com', 'icicibank.com', 'axisbank.com',
    'sbi.co.in', 'bankofamerica.com', 'chase.com', 'wellsfargo.com',
    'kotak.com', 'yesbank.in', 'idfcfirstbank.com',
    
    // Tech
    'microsoft.com', 'apple.com', 'ibm.com', 'oracle.com',
    'adobe.com', 'intel.com', 'nvidia.com', 'amd.com',
    'cisco.com', 'dell.com', 'hp.com', 'salesforce.com',
    
    // News
    'cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com',
    'nytimes.com', 'wsj.com', 'economictimes.com',
    
    // Other
    'yahoo.com', 'bing.com', 'duckduckgo.com', 'outlook.com',
    'office.com', 'protonmail.com', 'stackoverflow.com',
    'wikipedia.org', 'github.io', 'medium.com',
    'netflix.com', 'spotify.com',
];

// ===================== BLACKLIST PATTERNS =====================

const BLACKLIST_PATTERNS = [
    // Suspicious TLDs (only abused ones)
    /\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review|click|link|site|tech|store|shop|space|pro|biz|info|gq|men|party|science|racing|bid|win|faith|accountant|date|trade|webcam|zip|mov|rest|sbs|aaa|wang)$/i,
    
    // Suspicious patterns
    /login.*\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review)/i,
    /verify.*\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review)/i,
    /secure.*\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review)/i,
    /account.*\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review)/i,
    
    // IP addresses
    /\d+\.\d+\.\d+\.\d+/,
    
    // Brand misuse with suspicious TLDs
    /paypal.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /amazon.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /apple.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /microsoft.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /bank.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /hdfc.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /sbi.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /icici.*\.(xyz|tk|ml|ga|cf|top|club)/i,
];

const BRANDS = ['paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook',
                'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yesbank',
                'flipkart', 'netflix', 'spotify', 'github'];

const SENSITIVE_WORDS = ['login', 'verify', 'secure', 'update', 'confirm',
                         'password', 'account', 'signin', 'authenticate',
                         'validate', 'verification', 'billing', 'payment'];

const SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd',
                    'ow.ly', 't.co', 'buff.ly', 'rebrand.ly', 'lnkd.in'];

// ===================== PRE-CHECK FUNCTION =====================

function preCheckUrl(url) {
    const urlLower = url.toLowerCase();
    
    // Step 1: Check whitelist (SAFE)
    for (const domain of WHITELIST) {
        if (urlLower.includes(domain) || urlLower.endsWith(domain)) {
            return { verdict: 'legitimate', confidence: 1.0, source: 'whitelist' };
        }
    }
    
    // Step 2: Check blacklist patterns (PHISHING)
    for (const pattern of BLACKLIST_PATTERNS) {
        if (pattern.test(url)) {
            return { verdict: 'phishing', confidence: 0.9, source: 'blacklist_pattern' };
        }
    }
    
    // Step 3: Check for brand misuse
    let hasBrand = false;
    let hasSensitive = false;
    
    for (const brand of BRANDS) {
        if (urlLower.includes(brand)) {
            hasBrand = true;
            break;
        }
    }
    
    for (const word of SENSITIVE_WORDS) {
        if (urlLower.includes(word)) {
            hasSensitive = true;
            break;
        }
    }
    
    if (hasBrand && hasSensitive) {
        let isOfficial = false;
        for (const brand of BRANDS) {
            if (urlLower.includes(`${brand}.com`) || urlLower.includes(`${brand}.co.in`) || urlLower.includes(`${brand}.in`)) {
                isOfficial = true;
                break;
            }
        }
        if (!isOfficial) {
            return { verdict: 'phishing', confidence: 0.85, source: 'brand_misuse' };
        }
    }
    
    // Step 4: Check for shortened URLs
    for (const shortener of SHORTENERS) {
        if (urlLower.includes(shortener)) {
            return { verdict: 'phishing', confidence: 0.65, source: 'shortened_url' };
        }
    }
    
    return { verdict: 'uncertain', confidence: 0, source: 'uncertain' };
}

// ===================== LISTENER =====================

chrome.webNavigation.onBeforeNavigate.addListener(
    async function(details) {
        if (details.frameId !== 0) return;
        if (details.url.startsWith('chrome://')) return;
        if (details.url.startsWith('chrome-extension://')) return;
        if (details.url.startsWith('edge://')) return;
        
        const url = details.url;
        const tabId = details.tabId;
        
        console.log('🔍 PhishingNet Checking:', url.substring(0, 60) + '...');
        
        try {
            // Step 1: Pre-check
            const preCheckResult = preCheckUrl(url);
            
            if (preCheckResult.verdict === 'phishing') {
                console.log('⚡ Pre-check: PHISHING');
                const result = {
                    verdict: 'phishing',
                    confidence: preCheckResult.confidence,
                    layer: 'local_precheck',
                    explanation: `Blocked by: ${preCheckResult.source}`
                };
                redirectToWarning(tabId, url, result);
                return;
            }
            
            if (preCheckResult.verdict === 'legitimate') {
                console.log('✅ Pre-check: SAFE');
                return;
            }
            
            // Step 2: Check cache
            const cached = await getFromCache(url);
            if (cached) {
                console.log('💾 Cache:', cached.verdict);
                if (cached.verdict === 'phishing') {
                    redirectToWarning(tabId, url, cached);
                }
                return;
            }
            
            // Step 3: API call
            console.log('🌐 Calling API...');
            const apiResult = await checkUrlWithAPI(url);
            await saveToCache(url, apiResult);
            
            if (apiResult.verdict === 'phishing') {
                redirectToWarning(tabId, url, apiResult);
            }
            
        } catch (error) {
            console.error('❌ PhishingNet Error:', error);
        }
    },
    { url: [{ schemes: ['http', 'https'] }] }
);

// ===================== API FUNCTION =====================

async function checkUrlWithAPI(url) {
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url })
        });
        
        if (!response.ok) throw new Error('API request failed');
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return { verdict: 'legitimate', confidence: 0 };
    }
}

// ===================== CACHE FUNCTIONS =====================

async function getFromCache(url) {
    const result = await chrome.storage.local.get(url);
    if (result[url]) {
        const cached = result[url];
        if (Date.now() - cached.timestamp < CACHE_EXPIRY) {
            return cached.data;
        }
    }
    return null;
}

async function saveToCache(url, data) {
    await chrome.storage.local.set({
        [url]: { timestamp: Date.now(), data: data }
    });
}

// ===================== REDIRECT =====================

function redirectToWarning(tabId, url, result) {
    const warningUrl = chrome.runtime.getURL('warning.html') +
        `?url=${encodeURIComponent(url)}` +
        `&confidence=${result.confidence || 0.9}` +
        `&explanation=${encodeURIComponent(result.explanation || result.source || 'Suspicious pattern detected')}` +
        `&details=${encodeURIComponent(JSON.stringify(result.details || [result.explanation || 'Blocked by PhishingNet']))}`;
    
    chrome.tabs.update(tabId, { url: warningUrl });
}

// ===================== MESSAGE LISTENER =====================

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'continueAnyway') {
        console.log('⚠️ User continued to:', message.url);
        // Allow navigation
        return true;
    }
    if (message.action === 'getStats') {
        sendResponse({ status: 'ok', version: '2.0.0' });
        return true;
    }
});

console.log('🛡️ PhishingNet Extension Loaded!');