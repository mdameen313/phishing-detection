// background.js - PhishingNet Service Worker

const API_URL = 'https://phishing-detection-api-j1dv.onrender.com/predict';
const CACHE_EXPIRY = 3600000; // 1 hour

// Temporary bypass memory (URL -> timestamp)
let bypassedUrls = {};

const WHITELIST = [
    'google.com', 'google.co.in', 'gmail.com', 'youtube.com', 'drive.google.com', 'docs.google.com',
    'facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'reddit.com', 'github.com', 'tiktok.com', 'whatsapp.com', 'telegram.org', 'discord.com', 'twitch.tv',
    'amazon.com', 'amazon.in', 'flipkart.com', 'walmart.com', 'target.com', 'shopify.com', 'etsy.com', 'ebay.com', 'alibaba.com', 'aliexpress.com',
    'paypal.com', 'hdfcbank.com', 'icicibank.com', 'axisbank.com', 'sbi.co.in', 'bankofamerica.com', 'chase.com', 'wellsfargo.com', 'kotak.com', 'yesbank.in', 'idfcfirstbank.com',
    'microsoft.com', 'apple.com', 'ibm.com', 'oracle.com', 'adobe.com', 'intel.com', 'nvidia.com', 'amd.com', 'cisco.com', 'dell.com', 'hp.com', 'salesforce.com',
    'cnn.com', 'bbc.com', 'reuters.com', 'bloomberg.com', 'nytimes.com', 'wsj.com', 'economictimes.com',
    'yahoo.com', 'bing.com', 'duckduckgo.com', 'outlook.com', 'office.com', 'protonmail.com', 'stackoverflow.com', 'wikipedia.org', 'medium.com', 'netflix.com', 'spotify.com'
];

const BLACKLIST_PATTERNS = [
    /\.(xyz|tk|ml|ga|cf|top|club|work|online|live|stream|download|review|click|link|site|tech|store|shop|space|pro|biz|info|gq|men|party|science|racing|bid|win|faith|accountant|date|trade|webcam|zip|mov|rest|sbs|aaa|wang|help|lat|vip)$/i,
    /login.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /verify.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /\d+\.\d+\.\d+\.\d+/,
    /paypal.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /amazon.*\.(xyz|tk|ml|ga|cf|top|club)/i,
    /bank.*\.(xyz|tk|ml|ga|cf|top|club)/i
];

const BRANDS = ['paypal', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'hdfc', 'icici', 'sbi', 'axis', 'kotak', 'yesbank', 'flipkart', 'netflix', 'spotify', 'github'];
const SENSITIVE_WORDS = ['login', 'verify', 'secure', 'update', 'confirm', 'password', 'account', 'signin', 'authenticate', 'validate', 'verification', 'billing', 'payment'];
const SHORTENERS = ['bit.ly', 'tinyurl', 'goo.gl', 'shorturl', 'is.gd', 'ow.ly', 't.co', 'buff.ly', 'rebrand.ly', 'lnkd.in'];

function preCheckUrl(url) {
    const urlLower = url.toLowerCase();
    
    for (const domain of WHITELIST) {
        if (urlLower.includes(domain) || urlLower.endsWith(domain)) {
            return { verdict: 'legitimate', confidence: 1.0, source: 'whitelist' };
        }
    }
    
    for (const pattern of BLACKLIST_PATTERNS) {
        if (pattern.test(url)) {
            return { verdict: 'phishing', confidence: 0.9, source: 'blacklist_pattern' };
        }
    }
    
    let hasBrand = false, hasSensitive = false;
    for (const brand of BRANDS) { if (urlLower.includes(brand)) { hasBrand = true; break; } }
    for (const word of SENSITIVE_WORDS) { if (urlLower.includes(word)) { hasSensitive = true; break; } }
    
    if (hasBrand && hasSensitive) {
        let isOfficial = false;
        for (const brand of BRANDS) {
            if (urlLower.includes(`${brand}.com`) || urlLower.includes(`${brand}.in`)) { isOfficial = true; break; }
        }
        if (!isOfficial) return { verdict: 'phishing', confidence: 0.85, source: 'brand_misuse' };
    }
    
    for (const shortener of SHORTENERS) {
        if (urlLower.includes(shortener)) return { verdict: 'phishing', confidence: 0.65, source: 'shortened_url' };
    }
    
    return { verdict: 'uncertain', confidence: 0, source: 'uncertain' };
}

chrome.webNavigation.onBeforeNavigate.addListener(
    async function(details) {
        if (details.frameId !== 0) return;
        if (details.url.startsWith('chrome://') || details.url.startsWith('chrome-extension://') || details.url.startsWith('edge://')) return;
        
        const url = details.url;
        const tabId = details.tabId;

        // === BYPASS CHECK ===
        // If the user just clicked "Continue Anyway" on the warning page, let it load.
        if (bypassedUrls[url] && (Date.now() - bypassedUrls[url] < 5000)) {
            console.log('⚠️ User bypassed:', url);
            return; // Allow navigation
        }
        
        try {
            const preCheckResult = preCheckUrl(url);
            
            if (preCheckResult.verdict === 'phishing') {
                const result = {
                    verdict: 'phishing',
                    confidence: preCheckResult.confidence,
                    layer: 'local_precheck',
                    explanation: `Blocked by local rule: ${preCheckResult.source}`,
                    details: [`Blocked by local rule: ${preCheckResult.source}`]
                };
                redirectToWarning(tabId, url, result);
                return;
            }
            
            if (preCheckResult.verdict === 'legitimate') return;
            
            const cached = await getFromCache(url);
            if (cached) {
                if (cached.verdict === 'phishing') redirectToWarning(tabId, url, cached);
                return;
            }
            
            const apiResult = await checkUrlWithAPI(url);
            await saveToCache(url, apiResult);
            
            if (apiResult.verdict === 'phishing') redirectToWarning(tabId, url, apiResult);
            
        } catch (error) {
            console.error('❌ Error:', error);
        }
    },
    { url: [{ schemes: ['http', 'https'] }] }
);

async function checkUrlWithAPI(url) {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000); 

        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url: url }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return await response.json();

    } catch (error) {
        return { verdict: 'legitimate', confidence: 0 };
    }
}

async function getFromCache(url) {
    const result = await chrome.storage.local.get(url);
    if (result[url] && (Date.now() - result[url].timestamp < CACHE_EXPIRY)) return result[url].data;
    return null;
}

async function saveToCache(url, data) {
    await chrome.storage.local.set({ [url]: { timestamp: Date.now(), data: data } });
}

function redirectToWarning(tabId, url, result) {
    const confidence = result.confidence !== undefined ? result.confidence : 0.85;
    let detailsArray = result.details;
    if (!detailsArray || !Array.isArray(detailsArray) || detailsArray.length === 0) {
        detailsArray = [result.explanation || 'Suspicious pattern detected'];
    }
    const detailsString = encodeURIComponent(JSON.stringify(detailsArray));
    
    const warningUrl = chrome.runtime.getURL('warning.html') +
        `?url=${encodeURIComponent(url)}` +
        `&confidence=${confidence}` +
        `&details=${detailsString}`;
    
    chrome.tabs.update(tabId, { url: warningUrl });
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === 'continueAnyway') {
        // Save the URL to the bypass list for 5 seconds
        bypassedUrls[message.url] = Date.now();
        console.log('✅ Added to bypass list:', message.url);
    }
    return true;
});