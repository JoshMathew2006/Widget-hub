import {HandLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let sock;
let reconnectDelay = 500;
let lastMsgTime = performance.now();

// every existing ws.send(...) in the file now goes through this guard
const ws = {
    send(msg){
        if (sock && sock.readyState === WebSocket.OPEN) sock.send(msg);
    }
};
let cursorHistory = [];         
const CLICK_LOOKBACK_MS = 200;

const TAP_WINDOW_MS = 200;
const TAP_DROP = 0.24;
const TAP_MAX = 0.24;
const RELEASE_RISE = 0.07;
const GRAB_RELEASE_GAP = 0.25;

let ratioHistory = [];
let pinchMinR = Infinity;

function cursorAt(msAgo) {
    const target = performance.now() - msAgo;
    for (let i = cursorHistory.length - 1; i >= 0; i--) {
        if (cursorHistory[i].t <= target) return cursorHistory[i];
    }
    return cursorHistory[0] || null;
}

function connect(){
    sock = new WebSocket("ws://localhost:8000/ws");
    sock.onopen = () => {
        console.log("WS OPEN");
        reconnectDelay = 850;
        lastMsgTime = performance.now();
    };
    sock.onmessage = handleMessage;
    sock.onclose = (e) => {
        console.warn("WS CLOSED", e.code, new Date().toLocaleTimeString());
        draggingId = null;
        setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 5000);
    };
    sock.onerror = () => console.warn("WS ERROR", new Date().toLocaleTimeString());
}

setInterval(() => {
    if (sock && sock.readyState === WebSocket.OPEN && performance.now() - lastMsgTime > 3000) {
        console.warn("watchdog: no server message for 3s, closing");
        sock.close();
    }
}, 1000);

document.addEventListener("visibilitychange", () => {
    console.log("tab visible:", !document.hidden, new Date().toLocaleTimeString());
});

let handLandmarker = null;
let latestHandLandmarks = [];
let latestWorldLandmarks = [];
const webcamVideo = document.createElement("video");
webcamVideo.style.display = "none";
document.body.appendChild(webcamVideo);


let gestureCursor = null;
let mousePos = {x: 0, y: 0};
let smoothedCursor = { x: 0, y: 0 };
let cursorLostFrames = 0;
const SMOOTHING = 0.25;
const CURSOR_HIDE_AFTER = 20;

let recentPositions = [];
const POSITION_HISTORY_SIZE = 3;


let frameCount = 0;
const DETECT_EVERY_N_FRAMES = 2;

let wasPinching = false;
let pinchStartTime = 0;
let pinchAction = null;

const DRAG_HOLD_MS = 450;      // hold this long and it becomes a drag
const DRAG_START_DIST = 70;    // or move this many px and it becomes a drag
let pinchStartPos = null;
let dragStarted = false;
let grabOffset = { x: 0, y: 0 };

let inputMode = "mouse";
try { inputMode = localStorage.getItem("inputMode") || "mouse"; } catch (e) {}
let handStream = null;
let landmarkerLoading = null;

function setInputStatus(text) {
    const el = document.getElementById("inputStatus");
    if (el) el.textContent = text;
}

function loadHandLandmarker() {
    if (!landmarkerLoading) {
        landmarkerLoading = (async () => {
            const vision = await FilesetResolver.forVisionTasks(
                "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
            );
            handLandmarker = await HandLandmarker.createFromOptions(vision, {
                baseOptions: {
                    modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    delegate: "GPU"
                },
                runningMode: "VIDEO",
                numHands: 1
            });
        })();
        landmarkerLoading.catch(() => { landmarkerLoading = null; }); // allow a retry
    }
    return landmarkerLoading;
}

async function startHandTracking() {
    setInputStatus("Starting camera");
    try{
        await loadHandLandmarker();
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if(inputMode !== "hand"){
            stream.getTracks().forEach(t => t.stop());
            return;
        }
        handStream = stream;
        webcamVideo.srcObject = stream;
        await webcamVideo.play();
        setInputStatus("Camera On");
    } catch (err) {
        console.error("Hand Tracking Failed" , err);
        setInputStatus(err.name === "NotAllowedError" ? "Camera blocked" : "Camera unavailable");
        setInputMode("mouse" , { keepStatus: true});
    }
    
}

function stopHandTracking(){
    endUiDrag();
    if(handStream) {
        handStream.getTracks().forEach(t => t.stop());
        handStream = null;
    }
    webcamVideo.pause();
    webcamVideo.srcObject = null;
    latestHandLandmarks = [];
    latestWorldLandmarks = [];
    gestureCursor = null;
    recentPositions = [];
    cursorHistory = [];
    ratioHistory = [];
    awaitingGrab = false;
    pinchActive = false;
    pinchCounter = 0;
    pointingFrameCount = 0;
    if(wasPinching) draggingId = null;
    wasPinching = false;
    pinchAction = null;
    dragStarted = false;
    if (gestureHoverEl) { gestureHoverEl.classList.remove("gesture-hover"); gestureHoverEl = null; }
}

function setInputMode(mode, opts = {}) {
    inputMode = mode;
    try { localStorage.setItem("inputMode" , mode); } catch (e) {}
    const toggle = document.getElementById("inputToggle");
    toggle.classList.toggle("hand", mode === "hand");
    toggle.querySelectorAll("button").forEach(btn => {
        const on = btn.dataset.mode === mode;
        btn.classList.toggle("active" , on);
        btn.setAttribute("aria-checked", on);
    });
    if (mode === "hand") {
        startHandTracking();
    } else{
        stopHandTracking();
        if (!opts.keepStatus) setInputStatus("");
    }

}

function dist3D(a, b){
    return Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
}

function fingerReach(hand, tipIdx, knuckleIdx){
    return dist3D(hand[0], hand[tipIdx]) / dist3D(hand[0], hand[knuckleIdx]);
}

const POINT_INDEX_MIN  = 1.40;   // lower = you can bend your pointer more
const POINT_OTHERS_MAX = 1.50;   // middle/ring/pinky must be curled at least this much
const POINT_MARGIN     = 0.20;   // pointer must reach this much further than the others

let gestureDebug = {};   // last pointing check, shown when GESTURE_DEBUG is on

function isPointing(hand){
    const index = fingerReach(hand, 8, 5);
    const others = Math.max(
        fingerReach(hand, 12, 9),
        fingerReach(hand, 16, 13),
        fingerReach(hand, 20, 17)
    );
    const fails = [];
    if (index < POINT_INDEX_MIN)            fails.push("pointer too bent");
    if (others > POINT_OTHERS_MAX)          fails.push("other fingers open");
    if (index - others < POINT_MARGIN)      fails.push("pointer not out far enough");
    gestureDebug = { index, others, fails };
    return fails.length === 0;
}

let pointingFrameCount = 0;
const POINTING_CONFIRM_FRAMES = 2;

function updatePointingState(hand){
    if(isPointing(hand)){
        pointingFrameCount = Math.min(pointingFrameCount + 1, POINTING_CONFIRM_FRAMES);
    }
    else{
        pointingFrameCount = Math.max(pointingFrameCount - 1, 0);
    }
    return pointingFrameCount >= POINTING_CONFIRM_FRAMES;
}

const PINCH_ENTER = 0.15;
const PINCH_INDEX_MIN = 0;   //pointer finger extension value for pinches
const PINCH_EXIT = 0.35;
const PINCH_CONFIRM = 1;
const RELEASE_CONFIRM = 1;
const GESTURE_DEBUG = false;

function handSize(hand){
    return dist3D(hand[0] , hand[9]);
}

function pinchRatio(hand){
    const thumb = hand[4];
    const gap = Math.min(
        dist3D(thumb, hand[7]),
        dist3D(thumb, hand[8])
    );
    return gap / handSize(hand);
}

let pinchActive = false;
let noiseMin = 1, noiseMax = 0, noiseT = 0, missT = 0;
let pinchCounter = 0;
const REARM_RATIO = 0.30;

function updatePinchState(hand) {
    const r = pinchRatio(hand);
    const now = performance.now();

    // remember recent readings so we can spot a quick close
    ratioHistory.push({ t: now, r });
    while (ratioHistory.length && now - ratioHistory[0].t > TAP_WINDOW_MS) ratioHistory.shift();
    const recentMax = Math.max(...ratioHistory.map(p => p.r));

    if (!pinchActive) {
        const fullyClosed = r < PINCH_ENTER;                              // the normal pinch
        const quickFlick  = r < TAP_MAX && recentMax - r >= TAP_DROP;     // a short, fast close
        const pointerOut = fingerReach(hand, 8, 5) >= PINCH_INDEX_MIN;
        pinchCounter = (pointerOut && (fullyClosed || quickFlick)) ? pinchCounter + 1 : 0;
        const reachNow = fingerReach(hand, 8, 5);
        const gateOk = typeof PINCH_INDEX_MIN === "undefined" || reachNow >= PINCH_INDEX_MIN;
        if (r < 0.18 && !(gateOk && (fullyClosed || quickFlick)) && now - missT > 300) {
            missT = now;
            console.log("NO PINCH |",
                "gap " + r.toFixed(2) + (fullyClosed ? " ok" : " too wide, need <" + PINCH_ENTER),
                "| drop " + (recentMax - r).toFixed(2) + (quickFlick ? " ok" : " too small, need >=" + TAP_DROP),
                "| reach " + reachNow.toFixed(2) + (gateOk ? " ok" : " too bent, need >=" + PINCH_INDEX_MIN));
        }
        if (pinchCounter >= PINCH_CONFIRM) {
            pinchActive = true; pinchCounter = 0; pinchMinR = r;
            console.log(`PINCH ON  gap ${r.toFixed(2)} ${fullyClosed ? "(closed)" : "(flick)"}`);
        }
    } else {
        pinchMinR = Math.min(pinchMinR, r);
        const holding = pinchAction === "grab" && draggingId !== null;
        const reopened = holding
            ? r > GRAB_RELEASE_GAP                                   // grabs: fingers clearly open
            : r > PINCH_EXIT || r - pinchMinR >= RELEASE_RISE;       // clicks: quick backoff, unchanged
        pinchCounter = reopened ? pinchCounter + 1 : 0;
        if (pinchCounter >= (holding ? 2 : RELEASE_CONFIRM)) {
            console.log(`PINCH OFF gap ${r.toFixed(2)} min ${pinchMinR.toFixed(2)} rise ${(r - pinchMinR).toFixed(2)} ${holding ? "grab" : "tap"} after ${Math.round(now - pinchStartTime)}ms`);
            pinchActive = false; pinchCounter = 0; ratioHistory = [];
        }
    }
    if (!pinchActive) {
        noiseMin = Math.min(noiseMin, r);
        noiseMax = Math.max(noiseMax, r);
        if (now - noiseT > 2000) {
            //console.log("pinch gap range, last 2s:", noiseMin.toFixed(2), "to", noiseMax.toFixed(2));
            noiseMin = 1; noiseMax = 0; noiseT = now;
        }
    }
    if (GESTURE_DEBUG) {
        const d = gestureDebug;
        const verdict = pinchActive ? "● PINCH"
            : d.fails?.length ? "✗ " + d.fails.join(", ")
            : "✓ pointing";
        setInputStatus(`gap ${r.toFixed(2)} (<${PINCH_ENTER}) · drop ${(recentMax - r).toFixed(2)} (≥${TAP_DROP}) · reach ${fingerReach(hand, 8, 5).toFixed(2)} (≥${PINCH_INDEX_MIN}) · ${pinchActive ? "PINCH" : "no pinch"}`);
    }
    return pinchActive;
}

function remapRange(value, inMin, inMax, outMin, outMax) {

    const clamped = Math.max(inMin, Math.min(inMax, value));


    return ((clamped - inMin) / (inMax - inMin)) * (outMax - outMin) + outMin;
}

function getCursorPosition(hand){
    const smoothedRaw = getSmoothedRawPosition(hand);
    const xRemapped = remapRange(smoothedRaw.x , 0.15, 0.85, 0, 1);
    const yRemapped = remapRange(smoothedRaw.y, 0.22, 0.70, 0, 1);

    const screenX = (1 - xRemapped) * canvas.width;
    const screenY = yRemapped * canvas.height;
    return {x: screenX , y: screenY};
}

function getSmoothedRawPosition(hand){
    const CURSOR_TIP_WEIGHT = 0.5;
    const tip = hand[8], knuckle = hand[5];
    recentPositions.push({
        x: knuckle.x + (tip.x - knuckle.x) * CURSOR_TIP_WEIGHT,
        y: knuckle.y + (tip.y - knuckle.y) * CURSOR_TIP_WEIGHT,
    });
    if(recentPositions.length > POSITION_HISTORY_SIZE){
        recentPositions.shift();
    }
    const avgX = recentPositions.reduce((sum, p) => sum + p.x, 0) / recentPositions.length;
    const avgY = recentPositions.reduce((sum, p) => sum + p.y, 0) / recentPositions.length;
    return {x: avgX, y: avgY};
}

function smoothPosition(target){
    const s = dragStarted ? 0.5 : SMOOTHING;   // follow the hand more tightly while dragging
    smoothedCursor.x += (target.x - smoothedCursor.x) * s;
    smoothedCursor.y += (target.y - smoothedCursor.y) * s;
    return {x: smoothedCursor.x, y: smoothedCursor.y};
}

function screenToWorld(pos){
    return{
        x: pos.x / getScaleX(),
        y: pos.y / getScaleY()
    };
}

function computeHoveredId(){
    if (draggingId !== null) return draggingId;   
    const candidates = [];
    if (inputMode === "mouse") candidates.push(mousePos);
    if (gestureCursor) candidates.push(gestureCursor);

    for (const pos of candidates){
        const worldPos = screenToWorld(pos);
        for(const id in latestState){
            const obj = latestState[id];
            const halfW = obj.size[0] / 2;
            const halfH = obj.size[1] / 2;
            if(
                worldPos.x >= obj.x - halfW && worldPos.x <= obj.x + halfW &&
                worldPos.y >= obj.y - halfH && worldPos.y <= obj.y + halfH
            ){
                return id;
            }
        }
    }
    return null;
}

const WORLD_WIDTH = 800;
const WORLD_HEIGHT = 600;

let latestState = {}
let awaitingGrab = false;
let draggingId = null;
let selectedId = null;
let editingText = null;
let noteEditors = {};
let chatEditors = {};

let bgColor = "#000000";
let bodyColor = "#00E5FF";
let buttonColor = "#00E5FF";

function worldToScreen(x, y) {
    return {
        left: x * getScaleX(),
        top: y * getScaleY()
    };
}

function getContrastingTextColor(hex){
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;  // standard perceived-brightness formula
    return brightness > 150 ? "#000000" : "#FFFFFF";
}

function updateNoteTheme(entry){
    const textColor = getContrastingTextColor(bgColor);
    entry.editorDiv.querySelector(".ql-editor")?.style.setProperty("color", textColor);
    const toolbar = entry.wrapper.querySelector(".ql-toolbar");
    if (toolbar) {

        toolbar.querySelectorAll(".ql-stroke").forEach(el => el.style.stroke = textColor);
        toolbar.querySelectorAll(".ql-fill").forEach(el => el.style.fill = textColor);
        toolbar.style.color = textColor;

    }
}

function drawBox(halfW, halfH, isHovered) {
    
    const glowColor = isHovered ? bgColor : bodyColor;

    ctx.fillStyle = hexToRgba(bodyColor, 0.15);
    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 15;
    ctx.fillRect(-halfW, -halfH, halfW * 2, halfH * 2);

    ctx.shadowColor = glowColor;
    ctx.shadowBlur = 20;
    ctx.strokeStyle = bodyColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);

    ctx.shadowBlur = 10;
    ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);

    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#B8FBFF";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(-halfW, -halfH, halfW * 2, halfH * 2);
}

function ensureChatWindow(obj_id){
    if(chatEditors[obj_id]) return chatEditors[obj_id];

    const container = document.getElementById("noteContainer");

    const wrapper = document.createElement("div");
    wrapper.style.position = "absolute";
    wrapper.style.pointerEvents = "auto";
    wrapper.style.overflow = "hidden";
    wrapper.style.display = "flex";
    wrapper.style.flexDirection = "column";
    wrapper.style.borderRadius = "8px";
    wrapper.style.backdropFilter = "blur(10px)";
    wrapper.style.webkitBackdropFilter = "blur(10px)";
    wrapper.style.fontFamily = "'Andromeda', sans-serif";
    wrapper.style.boxSizing = "border-box";
    wrapper.style.minWidth = "280px";
    wrapper.style.minHeight = "250px";
    container.appendChild(wrapper);

    const accentColor = getContrastingTextColor(bgColor);


    const header = document.createElement("div");
    header.textContent = "M1L0";
    header.style.color = accentColor;
    header.style.padding = "8px 15px";
    header.style.fontSize = "18px";
    header.style.cursor = "grab";
    header.style.boxSizing = "border-box";
    header.style.fontFamily = "'Chakra Petch', sans-serif";
    header.style.userSelect = "none";
    wrapper.appendChild(header);

    const messagesEl = document.createElement("div");
    messagesEl.style.flex = "1";
    messagesEl.className = "chat-scrollable";
    messagesEl.style.overflow = "auto";
    messagesEl.style.padding = "8px";
    messagesEl.style.display = "flex";
    messagesEl.style.flexDirection = "column";
    messagesEl.style.gap = "6px";
    messagesEl.style.boxSizing = "border-box";
    messagesEl.style.userSelect = "none";
    wrapper.appendChild(messagesEl);

    const inputRow = document.createElement("div");
    inputRow.style.display = "flex";
    inputRow.style.padding = "8px";
    inputRow.style.gap = "6px";
    inputRow.style.boxSizing = "border-box";
    wrapper.appendChild(inputRow);



    const inputEl = document.createElement("input");
    inputEl.type = "text";
    inputEl.placeholder = "Message M1L0...";
    inputEl.style.flex = "1";
    inputEl.style.borderRadius = "4px";
    inputEl.style.border = "1px solid #ccc";
    inputEl.style.padding = "4px 8px";
    inputEl.style.boxSizing = "border-box";
    inputEl.style.fontFamily = "'Chakra Petch', sans-serif";
    inputRow.appendChild(inputEl);

    const sendBtn = document.createElement("button");
    sendBtn.textContent = "Send";
    sendBtn.style.cursor = "pointer";
    sendBtn.style.background = darkenColor(bodyColor, 90);
    sendBtn.style.border = "2px solid" + bodyColor;
    sendBtn.style.borderRadius = "4px";
    inputRow.appendChild(sendBtn);

    function sendMessage(){
        console.log("sendmessagecalled");
        const text = inputEl.value.trim();
        if(!text) return;
        ws.send(JSON.stringify({type: "chat_message", obj_id, text}));
        inputEl.value = "";
        showTypingIndicator(obj_id);
    }

    sendBtn.addEventListener("click", sendMessage);
    inputEl.addEventListener("keydown", (e) => {
        if (e.key === "Enter") sendMessage();
        e.stopPropagation();
    });

    wrapper.addEventListener("mousedown", (e) => {
        draggingId = obj_id;
        selectedId = obj_id;
        e.stopPropagation();
    });


    header.addEventListener("dblclick", (e) => {
        const factor = e.shiftKey ? 0.8 : 1.2;
        ws.send(JSON.stringify({ type: "resize_by_id", obj_id, factor }));
        e.stopPropagation();
    });

    chatEditors[obj_id] = {wrapper, messagesEl, sendBtn, lastMessageCount: 0};
    return chatEditors[obj_id];

}

function darkenColor(hex, amount){
    const r = Math.max(0, parseInt(hex.slice(1, 3), 16) - amount);
    const g = Math.max(0, parseInt(hex.slice(3, 5), 16) - amount);
    const b = Math.max(0, parseInt(hex.slice(5, 7), 16) - amount);
    return `rgb(${r}, ${g}, ${b})`;
}

function renderChatMessages(entry, messages){
    if(!messages || messages.length === entry.lastMessageCount) return;
    entry.messagesEl.innerHTML = "";
    for (const msg of messages) {
        const bubble = document.createElement("div");
        bubble.textContent = msg.text;
        bubble.dataset.role = msg.role;
        bubble.style.padding = "6px 10px";
        bubble.style.borderRadius = "6px";
        bubble.style.maxWidth = "80%";
        bubble.style.alignSelf = msg.role === "user" ? "flex-end" : "flex-start";
        bubble.style.background = msg.role === "user" ? hexToRgba(bodyColor, 0.4) : "rgba(255,255,255,0.1)";
        bubble.style.color = getContrastingTextColor(bgColor);
        bubble.style.fontFamily =  "'Chakra Petch', sans-serif";
        entry.messagesEl.appendChild(bubble);
    }
    const lastMsg = messages[messages.length - 1];   
    if (lastMsg && lastMsg.role === "user") {          
        appendTypingIndicator(entry);                   
    }                                                   

    entry.messagesEl.scrollTop = entry.messagesEl.scrollHeight;
    entry.lastMessageCount = messages.length;
}



function appendTypingIndicator(entry) {
    const dots = document.createElement("div");
    dots.className = "typing-indicator";
    dots.textContent = "...";
    dots.style.padding = "6px 10px";
    dots.style.alignSelf = "flex-start";
    dots.style.color = getContrastingTextColor(bgColor);
    dots.style.fontFamily = "'Chakra Petch', sans-serif";
    entry.messagesEl.appendChild(dots);
    entry.messagesEl.scrollTop = entry.messagesEl.scrollHeight;
}

function showTypingIndicator(obj_id){
    const entry = chatEditors[obj_id];
    if (!entry) return;
    appendTypingIndicator(entry);
}

// ---------- Weather widget ----------

const WEATHER_REFRESH_MS = 15 * 60 * 1000;
const WEATHER_RETRY_MS = 60 * 1000;
const DIAL_UPDATE_MS = 60 * 1000;   // moves the sun along the dial between fetches
const WEATHER_CODES = {
    0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain", 66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Showers", 81: "Showers", 82: "Violent showers", 85: "Snow showers", 86: "Snow showers",
    95: "Thunderstorm", 96: "Thunderstorm, hail", 97: "Heavy thunderstorm", 99: "Thunderstorm, hail",
};

function weatherIconKind(code) {
    if (code <= 1) return "clear";
    if (code === 2) return "partly";
    if (code === 3) return "cloudy";
    if (code === 45 || code === 48) return "fog";
    if (code >= 95) return "storm";
    if ((code >= 71 && code <= 77) || code === 85 || code === 86) return "snow";
    return "rain";   // drizzle, rain, freezing rain, showers
}

const ICON_SUN   = `<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>`;
const ICON_MOON  = `<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>`;
const ICON_CLOUD = `<path d="M7 15a4 4 0 0 1-.6-7.96A5.5 5.5 0 0 1 17 8a3.5 3.5 0 0 1 .5 7H7z"/>`;
const WEATHER_ICONS = {
    "clear-day":    ICON_SUN,
    "clear-night":  ICON_MOON,
    "partly-day":   `<g transform="translate(9 -1) scale(0.55)" stroke-width="2.8">${ICON_SUN}</g><g transform="translate(-1 4)">${ICON_CLOUD}</g>`,
    "partly-night": `<g transform="translate(10 0) scale(0.5)" stroke-width="3">${ICON_MOON}</g><g transform="translate(-1 4)">${ICON_CLOUD}</g>`,
    cloudy: `<g transform="translate(0 3)">${ICON_CLOUD}</g>`,
    fog:    `<path d="M4 8h16M6 12h12M4 16h16"/>`,
    rain:   `${ICON_CLOUD}<path d="M8 18l-1 3M12 18l-1 3M16 18l-1 3"/>`,
    snow:   `${ICON_CLOUD}<path d="M8 19h.01M12 21h.01M16 19h.01" stroke-width="2.6"/>`,
    storm:  `${ICON_CLOUD}<path d="M12.5 16l-2 3.5h3l-2 3.5"/>`,
};

// Arc from sunrise (left) to sunset (right); centre (50,52), radius 40
const DIAL_SVG = `
<svg viewBox="0 0 100 62" fill="none" stroke="currentColor" stroke-linecap="round" style="width:100%; display:block; overflow:visible">
    <path d="M4 52H96" stroke-width="1" opacity="0.35"/>
    <path d="M10 52A40 40 0 0 1 90 52" stroke-width="1.5" stroke-dasharray="2 3" opacity="0.45"/>
    <path class="progress" d="M10 52A40 40 0 0 1 90 52" pathLength="100" stroke-width="2.5" stroke-dasharray="0 100" style="display:none"/>
    <circle class="sun" cx="10" cy="52" r="4.5" fill="currentColor" stroke="none" style="display:none; filter: drop-shadow(0 0 3px currentColor)"/>
</svg>`;

let weatherEditors = {};
let geoPromise = null;   // one location request shared by every weather widget

function getUserLocation() {
    if (!geoPromise) {
        geoPromise = new Promise((resolve) => {
            if (!navigator.geolocation) return resolve(null);
            navigator.geolocation.getCurrentPosition(
                pos => resolve({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
                err => { console.warn("geolocation failed:", err.message); resolve(null); },
                { timeout: 10000, maximumAge: 30 * 60 * 1000 }
            );
        });
    }
    return geoPromise;
}

function weatherKey(loc) {
    return `${(+loc.lat).toFixed(3)},${(+loc.lon).toFixed(3)}`;
}

function makeDiv(css, text = "") {
    const d = document.createElement("div");
    d.style.cssText = css;
    d.textContent = text;
    return d;
}

function ensureWeatherWidget(obj_id) {
    if (weatherEditors[obj_id]) return weatherEditors[obj_id];

    const wrapper = makeDiv(`
        position: absolute; pointer-events: auto; overflow: hidden;
        display: flex; flex-direction: column; box-sizing: border-box;
        border-radius: 8px; backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        container-type: size; user-select: none; font-family: 'Chakra Petch', sans-serif;`);
    // cqh/cqw units scale with the widget, so everything stays proportional at any size
    const labelEl    = makeDiv("padding: 4cqh 6cqw 0; font-size: 9cqh; cursor: grab; opacity: 0.85; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;");
    const body       = makeDiv("flex: 1; min-height: 0; display: flex; align-items: center; gap: 4cqw; padding: 0 6cqw 4cqh;");

    const left       = makeDiv("flex: 1; min-width: 0; display: flex; flex-direction: column;");
    const tempEl     = makeDiv("font-family: 'Orbitron', sans-serif; font-size: 28cqh; line-height: 1;", "--°");
    const condEl     = makeDiv("display: flex; align-items: center; gap: 2cqw; font-size: 7.5cqh; margin-top: 2cqh; white-space: nowrap;");
    const iconEl     = makeDiv("width: 11cqh; height: 11cqh; flex: none;");
    const condTextEl = makeDiv("overflow: hidden; text-overflow: ellipsis;", "Locating…");
    const hiloEl     = makeDiv("font-size: 6.5cqh; opacity: 0.85; margin-top: 1.5cqh; white-space: nowrap;");
    const feelsEl    = makeDiv("font-size: 6.5cqh; opacity: 0.85; white-space: nowrap;");

    const right      = makeDiv("width: 34%; flex: none; display: flex; flex-direction: column; align-items: center; gap: 1cqh;");
    const dialEl     = makeDiv("width: 100%;");
    dialEl.innerHTML = DIAL_SVG;
    const sunLabelEl = makeDiv("font-size: 6cqh; opacity: 0.85; white-space: nowrap;");

    condEl.append(iconEl, condTextEl);
    left.append(tempEl, condEl, hiloEl, feelsEl);
    right.append(dialEl, sunLabelEl);
    body.append(left, right);
    wrapper.append(labelEl, body);
    document.getElementById("noteContainer").appendChild(wrapper);

    wrapper.addEventListener("mousedown", (e) => {
        draggingId = obj_id;
        selectedId = obj_id;
        e.stopPropagation();
    });
    labelEl.addEventListener("dblclick", (e) => {
        const factor = e.shiftKey ? 0.8 : 1.2;
        ws.send(JSON.stringify({ type: "resize_by_id", obj_id, factor }));
        e.stopPropagation();
    });

    const entry = {
        wrapper, labelEl, tempEl, iconEl, condTextEl, hiloEl, feelsEl, sunLabelEl,
        progressEl: dialEl.querySelector(".progress"),
        sunEl: dialEl.querySelector(".sun"),
        locating: true, inFlight: false, fetchedKey: null, fetchedAt: -Infinity,
        iconKind: null, sun: null, dialAt: 0,
    };
    weatherEditors[obj_id] = entry;

    getUserLocation().then(loc => {
        entry.locating = false;
        if (!loc) return;   // denied or failed: draw() fetches the server's default location
        const content = { label: "Your location", lat: loc.lat, lon: loc.lon };
        ws.send(JSON.stringify({ type: "set_location", obj_id, ...content }));
        fetchWeather(entry, content);   // don't wait for the server to echo it back
    });
    return entry;
}

function updateSunDial(entry) {
    const s = entry.sun;
    if (!s) return;
    const now = Date.now() / 1000;
    const fmt = t => new Date(t * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    const isDay = now >= s.rise && now < s.set;

    if (isDay) {
        const p = (now - s.rise) / (s.set - s.rise);   // 0 at sunrise, 1 at sunset
        entry.progressEl.setAttribute("stroke-dasharray", `${(p * 100).toFixed(1)} 100`);
        entry.sunEl.setAttribute("cx", (50 - 40 * Math.cos(Math.PI * p)).toFixed(1));
        entry.sunEl.setAttribute("cy", (52 - 40 * Math.sin(Math.PI * p)).toFixed(1));
        entry.sunLabelEl.textContent = `Sunset ${fmt(s.set)}`;
    } else {
        const nextRise = now < s.rise ? s.rise : s.riseNext;   // before dawn vs. after sunset
        entry.sunLabelEl.textContent = `Sunrise ${fmt(nextRise)}`;
    }
    entry.progressEl.style.display = isDay ? "" : "none";
    entry.sunEl.style.display = isDay ? "" : "none";
    entry.dialAt = performance.now();
}

async function fetchWeather(entry, loc) {
    entry.inFlight = true;
    entry.fetchedKey = weatherKey(loc);
    entry.fetchedAt = performance.now();
    try {
        const url = "https://api.open-meteo.com/v1/forecast"
            + `?latitude=${loc.lat}&longitude=${loc.lon}`
            + "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,is_day"
            + "&daily=temperature_2m_max,temperature_2m_min,sunrise,sunset&forecast_days=2"
            + "&temperature_unit=fahrenheit&wind_speed_unit=mph&timezone=auto&timeformat=unixtime";
        const res = await fetch(url);
        const data = await res.json();
        if (!res.ok || data.error) throw new Error(data.reason || `HTTP ${res.status}`);

        const c = data.current, d = data.daily;
        entry.tempEl.textContent = `${Math.round(c.temperature_2m)}°`;
        entry.condTextEl.textContent = WEATHER_CODES[c.weather_code] ?? "Unknown";
        entry.hiloEl.textContent = `H ${Math.round(d.temperature_2m_max[0])}° · L ${Math.round(d.temperature_2m_min[0])}°`;
        entry.feelsEl.textContent = `Feels ${Math.round(c.apparent_temperature)}° · ${Math.round(c.wind_speed_10m)} mph`;

        let kind = weatherIconKind(c.weather_code);
        if (kind === "clear" || kind === "partly") kind += c.is_day ? "-day" : "-night";
        if (entry.iconKind !== kind) {
            entry.iconEl.innerHTML =
                `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" style="width:100%; height:100%; display:block">${WEATHER_ICONS[kind]}</svg>`;
            entry.iconKind = kind;
        }

        entry.sun = { rise: d.sunrise[0], set: d.sunset[0], riseNext: d.sunrise[1] };
        updateSunDial(entry);
    } catch (err) {
        console.warn("weather fetch failed:", err);
        if (!entry.sun) entry.condTextEl.textContent = "Couldn't load, retrying";   // keep old data if we had some
        entry.fetchedAt = performance.now() - WEATHER_REFRESH_MS + WEATHER_RETRY_MS;
    } finally {
        entry.inFlight = false;
    }
}

function maybeFetchWeather(entry, loc) {
    if (entry.sun && performance.now() - entry.dialAt > DIAL_UPDATE_MS) updateSunDial(entry);
    if (!loc || entry.locating || entry.inFlight) return;
    const stale = performance.now() - entry.fetchedAt > WEATHER_REFRESH_MS;
    if (weatherKey(loc) !== entry.fetchedKey || stale) fetchWeather(entry, loc);
}

function cleanupRemovedWeather() {
    for (const id in weatherEditors) {
        if (!latestState[id]) {
            weatherEditors[id].wrapper.remove();
            delete weatherEditors[id];
        }
    }
}

function ensureNoteEditor(obj_id) {
    if (noteEditors[obj_id]) return noteEditors[obj_id].quill;

    const container = document.getElementById("noteContainer");

    const wrapper = document.createElement("div");
    wrapper.style.position = "absolute";
    wrapper.style.pointerEvents = "auto";
    wrapper.style.overflow = "hidden";
    wrapper.style.display = "flex";
    
    wrapper.style.flexDirection = "column";
    wrapper.style.borderRadius = "8px";
    wrapper.style.backdropFilter = "blur(10px)";
    wrapper.style.webkitBackdropFilter = "blur(10px)";

    container.appendChild(wrapper);

    const editorDiv = document.createElement("div");
    editorDiv.style.flex = "1";


    editorDiv.style.overflow = "auto";

    editorDiv.style.fontFamily = "'Andromeda', sans-serif";
    wrapper.appendChild(editorDiv);

    const quill = new Quill(editorDiv, { theme: "snow" });

    wrapper.addEventListener("mousedown", (e) => {
        selectedId = obj_id;

        editingText = null;
        e.stopPropagation();
    });

    
    const toolbarEl = wrapper.querySelector(".ql-toolbar");

    toolbarEl.style.fontFamily = "'Andromeda', sans-serif";  

    wrapper.querySelectorAll(".ql-picker-options").forEach(el => {
        el.style.backgroundColor = "#000000";

    });
    wrapper.querySelectorAll(".ql-picker-item").forEach(el => el.style.color = "#FFFFFF");

    wrapper.querySelectorAll(".ql-picker-label").forEach(el => el.style.color = "#FFFFFF");
    toolbarEl.style.borderTopLeftRadius = "6px";

    toolbarEl.style.borderTopRightRadius = "6px";

    editorDiv.style.borderBottomLeftRadius = "6px";


    editorDiv.style.borderBottomRightRadius = "6px";
    toolbarEl.addEventListener("mousedown", (e) => {
        draggingId = obj_id;
        selectedId = obj_id;
        e.preventDefault();
        e.stopPropagation();
    });

    toolbarEl.addEventListener("dblclick", (e) => {
        const factor = e.shiftKey ? 0.8 : 1.2;
        ws.send(JSON.stringify({ type: "resize_by_id", obj_id, factor }));
        e.stopPropagation();
    });

    noteEditors[obj_id] = { wrapper, quill, editorDiv };
    updateNoteTheme(noteEditors[obj_id]);
    return quill;

}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const bgPicker = new iro.ColorPicker("#bgColorPicker", {
    width: 100,
    color: "#000000"
});

const bodyPicker = new iro.ColorPicker("#bodyColorPicker", {
    width: 100,
    color: "#00E5FF"
});

const buttonPicker = new iro.ColorPicker("#buttonColorPicker", {
    width: 100,
    color: "#00E5FF"
});

function handleMessage(event) {
    lastMsgTime = performance.now();
    const data = JSON.parse(event.data);
    if (data.type === "state") {
        latestState = data.objects;
    }
    else if (data.type === "hit_test_result") {
        console.log("  hit ->", data.obj_id);
        if (inputMode === "hand"){
            if (!awaitingGrab) return;          
            awaitingGrab = false;
            if (!wasPinching) {                 
                selectedId = data.obj_id;
                return;
            }
        }
        draggingId = data.obj_id;
        selectedId = data.obj_id;
    }
    else if (data.type === "theme") {
        bgPicker.color.hexString = data.bg;
        bodyPicker.color.hexString = data.body;
        buttonPicker.color.hexString = data.button;
    }
    
    
}

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function getScaleX() {
    return canvas.width / WORLD_WIDTH;
}

function getScaleY() {
    return canvas.height / WORLD_HEIGHT;
}

function cleanupRemovedNotes(){
    for(const id in noteEditors){
        if(!latestState[id]){
            const entry = noteEditors[id];
            const emptyClone = entry.wrapper.cloneNode(false);
            entry.wrapper.replaceWith(emptyClone);
            emptyClone.remove()
            entry.quill = null;
            delete noteEditors[id];
        }
    }
}

function cleanupRemovedChats(){
    for(const id in chatEditors)
    {
        if(!latestState[id]){
            const entry = chatEditors[id];
            const emptyClone = entry.wrapper.cloneNode(false);
            entry.wrapper.replaceWith(emptyClone);
            emptyClone.remove()
            delete chatEditors[id]
        }
        
    }
}

function raiseWidget(id) {
    if (id === null) return;
    const entry = noteEditors[id] || chatEditors[id] || weatherEditors[id];
    const container = document.getElementById("noteContainer");
    if (entry && container.lastElementChild !== entry.wrapper) container.appendChild(entry.wrapper);
}

// Only touch the page when a widget actually moved or resized (or the window did).
// Writing the same styles every frame still makes the browser re-check the layout.
function placeWidget(entry, obj) {
    const key = `${obj.x},${obj.y},${obj.size[0]},${obj.size[1]},${canvas.width},${canvas.height}`;
    if (entry.lastPlace === key) return;
    const pos = worldToScreen(obj.x - obj.size[0] / 2, obj.y - obj.size[1] / 2);
    entry.wrapper.style.left = `${pos.left}px`;
    entry.wrapper.style.top = `${pos.top}px`;
    entry.wrapper.style.width = `${obj.size[0] * getScaleX()}px`;
    entry.wrapper.style.height = `${obj.size[1] * getScaleY()}px`;
    entry.lastPlace = key;
}

function draw() {

    frameCount++;
    let newDetection = false;
    if (inputMode === "hand" && handLandmarker && handStream && webcamVideo.readyState >= 2 && frameCount % DETECT_EVERY_N_FRAMES === 0){
        const results = handLandmarker.detectForVideo(webcamVideo, performance.now());
        latestHandLandmarks = results.landmarks;
        latestWorldLandmarks = results.worldLandmarks || [];
        newDetection = true;
    }

    let smoothedScreenPos = null;

    if (latestHandLandmarks.length > 0) {
        const hand = latestHandLandmarks[0];

        smoothedScreenPos = smoothPosition(getCursorPosition(hand));
        cursorHistory.push({ t: performance.now(), x: smoothedScreenPos.x, y: smoothedScreenPos.y });
        while (cursorHistory.length && performance.now() - cursorHistory[0].t > 1000) cursorHistory.shift();
        
        const shape = latestWorldLandmarks[0] || hand;
        const confirmed = (newDetection ? updatePointingState(shape) : pointingFrameCount >= POINTING_CONFIRM_FRAMES) || wasPinching;
        if (confirmed) {
            if (cursorLostFrames > 0) console.log("cursor recovered");
            cursorLostFrames = 0;
            gestureCursor = smoothedScreenPos;
        } else {
            cursorLostFrames++;
            if (cursorLostFrames === CURSOR_HIDE_AFTER + 1) {
                console.log("cursor HIDDEN (hand seen, not pointing)");
            }
        }
    } else {
        cursorLostFrames++;
        if (cursorLostFrames === CURSOR_HIDE_AFTER + 1) {
            console.log("cursor HIDDEN (no hand detected at all)");
        }
    }
    if (cursorLostFrames > CURSOR_HIDE_AFTER) {
        gestureCursor = null;
    }

    if (latestHandLandmarks.length > 0 && smoothedScreenPos) {
        const hand = latestHandLandmarks[0];
        const pinching = newDetection ? updatePinchState(latestWorldLandmarks[0] || hand) : pinchActive;
        const worldPos = screenToWorld(smoothedScreenPos);

        if (pinching && !wasPinching) {
            pinchStartTime = performance.now();
            pinchStartPos = { ...smoothedScreenPos };
            dragStarted = false;
            const back = cursorAt(CLICK_LOOKBACK_MS);
            const stable = back && Math.hypot(back.x - smoothedScreenPos.x, back.y - smoothedScreenPos.y) < 140;
            const clickPos = stable ? back : smoothedScreenPos;   // moving fast: grab what's under the cursor now
            console.log("  drift", back ? Math.round(Math.hypot(back.x - smoothedScreenPos.x, back.y - smoothedScreenPos.y)) : "n/a", stable ? "(used earlier spot)" : "(used current spot)");
            pinchAction = gestureClickAt(clickPos.x, clickPos.y, smoothedScreenPos);
            console.log("  ->", pinchAction);
        }
        else if (pinching && wasPinching) {
            if (pinchAction === "grab" && draggingId !== null) {
                if (!dragStarted) {
                    const held = performance.now() - pinchStartTime;
                    const moved = Math.hypot(smoothedScreenPos.x - pinchStartPos.x, smoothedScreenPos.y - pinchStartPos.y);
                    if (held >= DRAG_HOLD_MS || moved >= DRAG_START_DIST) {
                        dragStarted = true;
                        const obj = latestState[draggingId];
                        if (obj) {
                            // never lock in an offset that puts the cursor outside the widget
                            const maxX = obj.size[0] / 2 * 0.7;
                            const maxY = obj.size[1] / 2 * 0.7;
                            const rawX = obj.x - worldPos.x, rawY = obj.y - worldPos.y;
                            grabOffset = {
                                x: Math.max(-maxX, Math.min(maxX, rawX)),
                                y: Math.max(-maxY, Math.min(maxY, rawY)),
                            };
                            console.log("drag start offset:", rawX.toFixed(0), rawY.toFixed(0), "-> clamped", grabOffset.x.toFixed(0), grabOffset.y.toFixed(0));
                        } else {
                            grabOffset = { x: 0, y: 0 };
                        }
                    }
                }
                if (dragStarted) {
                    ws.send(JSON.stringify({ type: "drag", obj_id: draggingId, x: worldPos.x + grabOffset.x, y: worldPos.y + grabOffset.y }));
                }
            } else if (pinchAction === "uidrag") {
                moveUiDrag(smoothedScreenPos);
            }
        }
        else if (!pinching && wasPinching) {
            if (pinchAction === "uidrag") endUiDrag();
            draggingId = null;
            pinchAction = null;
            dragStarted = false;
        }

        wasPinching = pinching;
    }

   
    const hoveredId = computeHoveredId();
    raiseWidget(draggingId);

    cleanupRemovedNotes();
    cleanupRemovedWeather();
    cleanupRemovedChats();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(getScaleX(), getScaleY());
    for (const id in latestState) {
        const obj = latestState[id];
        const halfW = obj.size[0] / 2;
        const halfH = obj.size[1] / 2;
        ctx.save();
        ctx.translate(obj.x, obj.y);
        ctx.rotate(obj.angle);
        ctx.scale(1 / getScaleX(), 1 / getScaleY());

        if (obj.type !== "note" && obj.type !== "chat" && obj.type !== "weather") {
            drawBox(halfW, halfH, id === hoveredId);
        }

        if (id === selectedId && obj.type !== "note" && obj.type !== "chat" && obj.type !== "weather") {
            ctx.shadowColor = "#FFFFFF";
            ctx.shadowBlur = 20;
            ctx.strokeStyle = "#FFFFFF";
            ctx.lineWidth = 3;
            const pad = 4
            ctx.strokeRect(-halfW - pad, -halfH - pad, obj.size[0] + pad * 2, obj.size[1] + pad * 2);
        }
        ctx.shadowBlur = 0;
        ctx.restore();

        if (obj.type === "note") {
            ensureNoteEditor(id);
            const entry = noteEditors[id];
            placeWidget(entry, obj);
            const outlineColor = (id === selectedId) ? "#FFFFFF" : (id === hoveredId ? bgColor : bodyColor);
            if (entry.lastOutline !== outlineColor) {
                entry.wrapper.style.background = hexToRgba(bodyColor, 0.25);
                entry.wrapper.style.border = `2px solid ${outlineColor}`;
                entry.wrapper.style.boxShadow = `
                    0 0 20px ${outlineColor},
                    0 0 10px ${outlineColor},
                    inset 0 0 0 1px #B8FBFF
                `;
                entry.lastOutline = outlineColor;

            }

        }
        else if (obj.type === "chat"){
            
            const entry = ensureChatWindow(id);
            placeWidget(entry, obj);

            const colorKey = bodyColor + bgColor;
            if (entry.lastColors !== colorKey) {          // only restyle when the theme changes
                entry.sendBtn.style.background = bodyColor;
                entry.sendBtn.style.border = bodyColor;
                const textColor = getContrastingTextColor(bgColor);
                Array.from(entry.messagesEl.children).forEach(bubble => {
                    if (bubble.className === "typing-indicator") return;
                    bubble.style.color = textColor;
                    if (bubble.dataset.role === "user") {
                        bubble.style.background = hexToRgba(bodyColor, 0.4);
                    }
                });
                entry.lastColors = colorKey;
                entry.lastOutline = null;                 // repaint the border with the new colors too
            }
            
            const outlineColor = (id === selectedId) ? "#FFFFFF" : (id === hoveredId ? bgColor : bodyColor);
            if (entry.lastOutline !== outlineColor) {
                entry.wrapper.style.background = hexToRgba(bodyColor, 0.25);
                entry.wrapper.style.border = `2px solid ${outlineColor}`;
                entry.wrapper.style.boxShadow = `
                    0 0 20px ${outlineColor},
                    0 0 10px ${outlineColor},
                    inset 0 0 0 1px #B8FBFF`;
                entry.lastOutline = outlineColor;

            }
            renderChatMessages(entry, obj.content);
        }
        else if (obj.type === "weather") {
            
            const entry = ensureWeatherWidget(id);
            placeWidget(entry, obj);

            const textColor = getContrastingTextColor(bgColor);
            if (entry.lastText !== textColor) {
                entry.wrapper.style.color = textColor;
                entry.lastText = textColor;
            }
            if (obj.content && entry.labelEl.textContent !== obj.content.label) {
                entry.labelEl.textContent = obj.content.label;
            }

            const outlineColor = (id === selectedId) ? "#FFFFFF" : (id === hoveredId ? bgColor : bodyColor);
            if (entry.lastOutline !== outlineColor) {
                entry.wrapper.style.background = hexToRgba(bodyColor, 0.25);
                entry.wrapper.style.border = `2px solid ${outlineColor}`;
                entry.wrapper.style.boxShadow = `
                    0 0 20px ${outlineColor},
                    0 0 10px ${outlineColor},
                    inset 0 0 0 1px #B8FBFF`;
                entry.lastOutline = outlineColor;
            }
            maybeFetchWeather(entry, obj.content);
        }


    }
    ctx.restore();
    

    ctx.shadowBlur = 0;

    const cursorEl = document.getElementById("gestureCursorEl");
    if(gestureCursor){
        cursorEl.style.display = "block";
        cursorEl.style.left = `${gestureCursor.x - 12}px`;
        cursorEl.style.top = `${gestureCursor.y - 12}px`;
        cursorEl.style.borderColor = bodyColor;  
    } else {
        cursorEl.style.display = "none";
    }

    updateHub();
    updateTrash();
    updateGestureHover();
    requestAnimationFrame(draw);
}


// ---------- Shared click targets (used by both mouse and hand input) ----------
const GESTURE = {
    clickable: "button, [data-gesture-click]",
    drag: "[data-gesture-drag]",
    ignore: "[data-gesture-ignore]",
    surface: "[data-gesture-surface]",
};

const canvasButtons = [];

function canvasButtonAt(x, y) {
    return canvasButtons.find(({ rect }) => {
        const r = rect();
        return x >= r.x && x < r.x + r.size && y >= r.y && y < r.y + r.size;
    }) || null;
}

// What HTML UI is under a point?
function uiAt(x, y) {
    const el = document.elementFromPoint(x, y);
    if (!el || el === canvas || el.closest(GESTURE.ignore)) {
        return { el: null, clickable: null, draggable: null, onSurface: false };
    }
    return {
        el,
        clickable: el.closest(GESTURE.clickable),
        draggable: el.closest(GESTURE.drag),
        onSurface: !!el.closest(GESTURE.surface),
    };
}

// ---------- Hand tracking: pinch-drag on sliders / color wheels ----------
let uiDrag = null;   // { el, startClick, startHand, last }

function fireMouse(target, type, x, y) {
    const init = {
        bubbles: true, cancelable: true, view: window,
        clientX: x, clientY: y, button: 0, buttons: type === "up" ? 0 : 1,
        pointerId: 1, pointerType: "mouse", isPrimary: true,
    };
    target.dispatchEvent(new PointerEvent("pointer" + type, init));
    target.dispatchEvent(new MouseEvent("mouse" + type, init));
}

function startUiDrag(el, x, y, handPos) {
    uiDrag = { el, startClick: { x, y }, startHand: { ...handPos }, last: { x, y } };
    fireMouse(el, "down", x, y);
}

function moveUiDrag(handPos) {
    if (!uiDrag) return;
    // move relative to where the pinch started, so closing your fingers doesn't jolt it
    const x = uiDrag.startClick.x + (handPos.x - uiDrag.startHand.x);
    const y = uiDrag.startClick.y + (handPos.y - uiDrag.startHand.y);
    uiDrag.last = { x, y };
    fireMouse(uiDrag.el, "move", x, y);
}

function endUiDrag() {
    if (!uiDrag) return;
    fireMouse(uiDrag.el, "up", uiDrag.last.x, uiDrag.last.y);
    uiDrag = null;
}

// ---------- Hand tracking: pinch to click ----------
function flashCursor() {
    const c = document.getElementById("gestureCursorEl");
    c.classList.add("click");
    setTimeout(() => c.classList.remove("click"), 150);
}

function gestureClickAt(x, y, handPos) {
    const canvasBtn = canvasButtonAt(x, y);
    if (canvasBtn) {
        canvasBtn.onPress();
        flashCursor();
        return "ui";
    }

    const { el, clickable, draggable, onSurface } = uiAt(x, y);
    if (draggable) {
        startUiDrag(el, x, y, handPos);
        flashCursor();
        return "uidrag";
    }
    if (clickable) {
        clickable.click();
        flashCursor();
        return "ui";
    }
    if (onSurface) return "ui";   // pinched a panel's background: ignore

    // Empty space or a widget: close settings and try to grab
    //document.getElementById("settingsPanel").style.display = "none";
    const w = screenToWorld({ x, y });
    awaitingGrab = true;
    ws.send(JSON.stringify({ type: "hit_test", x: w.x, y: w.y }));
    return "grab";
}

// Glow on whichever clickable/draggable the hand cursor is over
let gestureHoverEl = null;
function updateGestureHover() {
    const ui = (gestureCursor && draggingId === null) ? uiAt(gestureCursor.x, gestureCursor.y) : null;  
    const target = ui ? (ui.clickable || ui.draggable) : null;
    if (target !== gestureHoverEl) {
        if (gestureHoverEl) gestureHoverEl.classList.remove("gesture-hover");
        if (target) target.classList.add("gesture-hover");
        gestureHoverEl = target;
    }
}

// ---------- Mouse click (shares the same targets) ----------
function handleClickAt(screenX, screenY){
    const canvasBtn = canvasButtonAt(screenX, screenY);
    if (canvasBtn) {
        canvasBtn.onPress();
        return;
    }

    const { clickable } = uiAt(screenX, screenY);
    if (clickable) {
        clickable.click();
        return;
    }

    const worldX = screenX / getScaleX();
    const worldY = screenY / getScaleY();
    ws.send(JSON.stringify({type: "hit_test", x: worldX, y: worldY}));
}


canvas.addEventListener("mousedown", (e) => {
    if (inputMode !== "mouse") return;
    handleClickAt(e.offsetX, e.offsetY);
});

document.addEventListener("mousemove", (e) => {
    if (inputMode !== "mouse") return;
    if (draggingId === null) return;

    const rect = canvas.getBoundingClientRect();
    const worldX = (e.clientX - rect.left) / getScaleX();
    const worldY = (e.clientY - rect.top) / getScaleY();

    ws.send(JSON.stringify({
        type: "drag",
        obj_id: draggingId,
        x: worldX,
        y: worldY
    }));
});

document.addEventListener("mouseup", (e) => {
    if (inputMode !== "mouse") return;
    draggingId = null;
});

canvas.addEventListener("dblclick", (e) => {
    if (inputMode !== "mouse") return;
    const worldX = e.offsetX / getScaleX();
    const worldY = e.offsetY / getScaleY();

    const factor = e.shiftKey ? 0.5 : 2.0;

    ws.send(JSON.stringify({
        type: "resize",
        x: worldX,
        y: worldY,
        factor: factor
    }));
});

canvas.addEventListener("mousemove" , (e) => {
    if (inputMode !== "mouse") return;
    mousePos = {x: e.offsetX, y: e.offsetY};
});



window.addEventListener("keydown", (e) => {
    if (selectedId === null) return;
    const obj = latestState[selectedId];
    if (!obj) return;

    if (obj.type === "note") return; // Quill handles its own typing

    if (e.key === "Delete" || e.key === "Backspace") {
        ws.send(JSON.stringify({ type: "remove", obj_id: selectedId }));
        selectedId = null;
    }
});

document.getElementById("settingsToggle").addEventListener("click", () => {
    const panel = document.getElementById("settingsPanel");
    panel.style.display = getComputedStyle(panel).display === "none" ? "flex" : "none";
});

document.querySelectorAll(".widget-option").forEach(btn => {
    btn.addEventListener("click", (e) => {
        const chosenType = btn.dataset.type;
        ws.send(JSON.stringify({ type: "spawn", window_type: chosenType }));
        document.getElementById("widgetPicker").style.display = "none";
    });
});

bgPicker.on("color:change", (color) => {
    bgColor = color.hexString;
    document.body.style.background = bgColor;
    document.getElementById("hub").style.setProperty("--hub-bg", bgColor);
    document.getElementById("settingsPanel").style.background = hexToRgba(bgColor, 0.35);
    document.getElementById("widgetPicker").style.background = hexToRgba(bgColor, 0.35);
    for (const id in noteEditors) {
        updateNoteTheme(noteEditors[id]);
    }
});

bodyPicker.on("color:change", (color) => {
    bodyColor = color.hexString;
});

buttonPicker.on("color:change", (color) => {
    buttonColor = color.hexString;
    document.getElementById("inputToggle").style.setProperty("--toggle", buttonColor);
    document.getElementById("hub").style.setProperty("--hub", buttonColor);
    trashEl.style.setProperty("--trash", buttonColor);

    const gearBtn = document.getElementById("settingsToggle");
    gearBtn.style.color = buttonColor;
    gearBtn.style.textShadow = `0 0 10px ${buttonColor}, 0 0 20px ${buttonColor}`;

    const panel = document.getElementById("settingsPanel");
    panel.style.border = `1px solid ${buttonColor}33`;
    panel.style.boxShadow = `0 0 12px ${buttonColor}44`;

    const labels = document.querySelectorAll("#settingsPanel p");
    labels.forEach(label => label.style.color = buttonColor);

    const widgetPicker = document.getElementById("widgetPicker");
    widgetPicker.style.border = `1px solid ${buttonColor}33`;
    widgetPicker.style.boxShadow = `0 0 12px ${buttonColor}44`;

    const widgetButtons = document.querySelectorAll(".widget-option");
    widgetButtons.forEach(btn => {
        btn.style.color = buttonColor;
        btn.style.border = `1px solid ${buttonColor}`;
        btn.style.textShadow = `0 0 6px ${buttonColor}`;
    });
});

// ---------- Radial add menu ----------
const HUB_CLOSED_REACH = 75;      // cursor this close to the center can open the menu...
const HUB_OPEN_DWELL_MS = 250;    // ...but only after resting there this long
const HUB_OPEN_RX = 445;          // while open, an oval just outside the bubbles:
const HUB_OPEN_RY = 260;          //   side bubbles reach ~415px out, top/bottom ~227px
const HUB_CLOSE_DELAY_MS = 300;   // grace time so crossing gaps doesn't close it

let hubOpen = false;
let hubCloseAt = 0;
let hubHoverSince = 0;            // when the cursor arrived near the center
let mouseClient = { x: -9999, y: -9999 };

document.addEventListener("mousemove", (e) => {
    mouseClient = { x: e.clientX, y: e.clientY };
});

function setHubOpen(open) {
    hubOpen = open;
    document.body.classList.toggle("hub-open", open);
}

// any bubble with a data-type spawns that widget when clicked
document.querySelectorAll("#hubBubbles .hub-bubble[data-type]").forEach(btn => {
    btn.addEventListener("click", () => {
        ws.send(JSON.stringify({ type: "spawn", window_type: btn.dataset.type }));
        setHubOpen(false);
    });
});

function updateHub() {
    const now = performance.now();
    const cx = canvas.width / 2, cy = canvas.height / 2;
    const points = [];
    if (inputMode === "mouse") points.push(mouseClient);
    if (gestureCursor) points.push(gestureCursor);

    const nearCenter = points.some(p => Math.hypot(p.x - cx, p.y - cy) < HUB_CLOSED_REACH);
    const insideMenu = points.some(p => ((p.x - cx) / HUB_OPEN_RX) ** 2 + ((p.y - cy) / HUB_OPEN_RY) ** 2 < 1);
    const blocked = draggingId !== null || trashActive;

    if (!hubOpen) {
        if (nearCenter && !blocked) {
            if (!hubHoverSince) hubHoverSince = now;
            if (now - hubHoverSince >= HUB_OPEN_DWELL_MS) {
                setHubOpen(true);
                hubCloseAt = now + HUB_CLOSE_DELAY_MS;
            }
        } else {
            hubHoverSince = 0;   // passed through without stopping
        }
        return;
    }

    if (insideMenu && !blocked) {
        hubCloseAt = now + HUB_CLOSE_DELAY_MS;
    } else if (now > hubCloseAt) {
        setHubOpen(false);
        hubHoverSince = 0;
    }
}



// ---------- Trash zone (bottom-left) ----------
const TRASH_ENTER_RADIUS = 230;     // cursor this close to the corner starts the timer
const TRASH_EXIT_RADIUS = 280;      // must leave this far to cancel, so the edge doesn't flicker
const TRASH_HOLD_MS = 500;         // how long the bar takes to fill
const TRASH_CANCEL_GRACE_MS = 150;  // brief slips outside the zone don't cancel

const trashEl = document.getElementById("trash");
const trashFill = document.querySelector("#trashBar .fill");
trashEl.style.setProperty("--trash-r", TRASH_ENTER_RADIUS + "px");   // the visible arc matches the entry radius

let trashActive = false;
let trashTarget = null;
let trashStart = 0;
let trashLeftAt = 0;
let trashLocked = false;   // after a delete, the cursor must leave the zone before it can trigger again

function setTrashActive(on) {
    trashActive = on;
    document.body.classList.toggle("trash-active", on);
    if (!on) {
        trashTarget = null;
        trashFill.setAttribute("stroke-dasharray", "0 100");
    }
}

function updateTrash() {
    const now = performance.now();
    const points = [];
    if (inputMode === "mouse") points.push(mouseClient);
    if (gestureCursor) points.push(gestureCursor);
    const dist = Math.min(Infinity, ...points.map(p => Math.hypot(p.x - canvas.width, p.y)));   
    const inside = dist < (trashActive ? TRASH_EXIT_RADIUS : TRASH_ENTER_RADIUS);
    if (!inside) trashLocked = false;

    const target = draggingId !== null ? draggingId : selectedId;
    const targetOk = target !== null && !!latestState[target];

    if (!trashActive) {
        if (inside && targetOk && !trashLocked && !hubOpen) {
            trashTarget = target;
            trashStart = now;
            trashLeftAt = 0;
            setTrashActive(true);
        }
        return;
    }

    if (target !== trashTarget || !targetOk) { setTrashActive(false); return; }

    if (!inside) {
        if (!trashLeftAt) trashLeftAt = now;
        if (now - trashLeftAt > TRASH_CANCEL_GRACE_MS) setTrashActive(false);
        return;
    }
    trashLeftAt = 0;

    const p = (now - trashStart) / TRASH_HOLD_MS;   // 0 -> 1 as the bar fills
    if (p >= 1) {
        ws.send(JSON.stringify({ type: "remove", obj_id: trashTarget }));
        if (selectedId === trashTarget) selectedId = null;
        if (draggingId === trashTarget) draggingId = null;
        trashLocked = true;
        setTrashActive(false);
        return;
    }
    trashFill.setAttribute("stroke-dasharray", `${(p * 100).toFixed(1)} 100`);
}

document.querySelectorAll("#inputToggle button").forEach(btn => {
    btn.addEventListener("click", () => {
        if (btn.dataset.mode !== inputMode) setInputMode(btn.dataset.mode);
    });
});
setInputMode(inputMode);

connect();
draw();

// tell the server when YOU change a color (input:end only fires for real drags,
// so colors arriving from the server don't get echoed back)
function sendTheme() {
    ws.send(JSON.stringify({ type: "theme", bg: bgColor, body: bodyColor, button: buttonColor }));
}
bgPicker.on("input:end", sendTheme);
bodyPicker.on("input:end", sendTheme);
buttonPicker.on("input:end", sendTheme);