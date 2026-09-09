const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const ws = new WebSocket("ws://localhost:8000/ws");

import {HandLandmarker, FilesetResolver} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

let handLandmarker = null;
let latestHandLandmarks = [];
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

let recentAngles = [];
const ANGLE_HISTORY_SIZE = 5;

let frameCount = 0;
const DETECT_EVERY_N_FRAMES = 2;

let wasPinching = false;
let pinchStartTime = 0;

async function initHandTracking(){
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numHands: 2
    });

    const stream = await navigator.mediaDevices.getUserMedia({ video: true});
    webcamVideo.srcObject = stream;
    await webcamVideo.play();
}

initHandTracking().catch(err => console.error("Hand tracking init failed:", err));

function getJointAngle(hand, baseIdx, midIdx, tipIdx){
    const base = hand[baseIdx];
    const mid = hand[midIdx];
    const tip = hand[tipIdx];

    const v1 = {x: base.x - mid.x, y: base.y - mid.y, z: base.z - mid.z };
    const v2 = {x: tip.x - mid.x, y: tip.y - mid.y , z: tip.z - mid.z};

    const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    const mag1 = Math.sqrt(v1.x**2 + v1.y**2 + v1.z**2);
    const mag2 = Math.sqrt(v2.x**2 + v2.y**2 + v2.z**2);

    const angleRad = Math.acos(dot / (mag1 * mag2));
    return angleRad * (180 / Math.PI);
}

function isFingerExtended(hand, tipIdx, baseIdx, threshold = 1.2){
    const wrist = hand[0];
    const tip = hand[tipIdx];
    const base = hand[baseIdx];

    const distTip = Math.hypot(tip.x - wrist.x , tip.y - wrist.y, tip.z - wrist.z);
    const distBase = Math.hypot(base.x - wrist.x, base.y - wrist.y, base.z - wrist.z);

    return distTip > distBase * threshold;
}

function isIndexExtended(hand){
    const angle = getJointAngle(hand, 5, 6, 8);
    return angle > 140;
}

function isPointing(hand){
    const indexExtended = isIndexExtended(hand);
    const middleExtended = isFingerExtended(hand, 12, 9, 1.2);
    const ringExtended = isFingerExtended(hand, 16, 13, 1.2);
    const pinkyExtended = isFingerExtended(hand, 20, 17, 1.2);
    return indexExtended && !middleExtended && !ringExtended && !pinkyExtended;
}


let pointingFrameCount = 0;
const POINTING_CONFIRM_FRAMES = 2;

function updatePointingState(hand){
    if(isPointing(hand)){
        pointingFrameCount = Math.min(pointingFrameCount + 1, POINTING_CONFIRM_FRAMES);
    }else{
        pointingFrameCount = Math.max(pointingFrameCount - 1, 0);
    }
    return pointingFrameCount >= POINTING_CONFIRM_FRAMES;
}

function getPinchDistance(hand){
    const thumbTip = hand[4];
    const indexTip = hand[8];
    return Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y, thumbTip.z - indexTip.z);

}

function isPinching(hand){
    return getPinchDistance(hand) < 0.06;
}

let pinchFrameCount = 0;
const PINCH_CONFIRM_FRAMES = 2;
function updatePinchState(hand){
    if(isPinching(hand)){
        pinchFrameCount = Math.min(pinchFrameCount + 1, PINCH_CONFIRM_FRAMES);

    }else{
        pinchFrameCount = Math.max(pinchFrameCount -1, 0);
    }
    return pinchFrameCount >= PINCH_CONFIRM_FRAMES;
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
    const indexTip = hand[8];
    recentPositions.push({x: indexTip.x, y: indexTip.y});
    if(recentPositions.length > POSITION_HISTORY_SIZE){
        recentPositions.shift();
    }
    const avgX = recentPositions.reduce((sum, p) => sum + p.x, 0) / recentPositions.length;
    const avgY = recentPositions.reduce((sum, p) => sum + p.y, 0) / recentPositions.length;
    return {x: avgX, y: avgY};
}

function smoothPosition(target){
    smoothedCursor.x += (target.x - smoothedCursor.x) * SMOOTHING;
    smoothedCursor.y += (target.y - smoothedCursor.y) * SMOOTHING;
    return {x: smoothedCursor.x, y: smoothedCursor.y};
}

function screenToWorld(pos){
    return{
        x: pos.x / getScaleX(),
        y: pos.y / getScaleY()
    };
}

function computeHoveredId(){
    const candidates = [mousePos];
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

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === "state") {
        latestState = data.objects;
    }
    else if (data.type === "hit_test_result") {
        console.log("got hit test result:", data.obj_id);
        draggingId = data.obj_id;
        selectedId = data.obj_id;
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

function draw() {

    frameCount++;
    if (handLandmarker && webcamVideo.readyState >= 2 && frameCount % DETECT_EVERY_N_FRAMES === 0){
        const results = handLandmarker.detectForVideo(webcamVideo, performance.now());
        latestHandLandmarks = results.landmarks;
    }


    if (latestHandLandmarks.length > 0) {
        const hand = latestHandLandmarks[0];
        console.log("index y:", hand[8].y.toFixed(3));
        const confirmed = updatePointingState(hand);
        if (confirmed) {
            if (cursorLostFrames > 0) console.log("cursor recovered");
            cursorLostFrames = 0;
            gestureCursor = smoothPosition(getCursorPosition(hand));
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

    if(latestHandLandmarks.length > 0 && gestureCursor){
        const hand = latestHandLandmarks[0];
        const pinching = updatePinchState(hand);
        const worldPos = screenToWorld(gestureCursor);

        if(pinching && !wasPinching){
            pinchStartTime = performance.now();
            console.log("PINCH START, sending hit_test at", worldPos);
            ws.send(JSON.stringify({type: "hit_test", x: worldPos.x, y: worldPos.y}));

        }
        else if(pinching && wasPinching){
            console.log("still pinching, draggingId:", draggingId);
            if(draggingId !== null){
                ws.send(JSON.stringify({type: "drag", obj_id: draggingId, x: worldPos.x, y: worldPos.y}));
            }
        }
        else if(!pinching && wasPinching){
            const elapsed = performance.now() - pinchStartTime;
            console.log("pinch ended, duration:", elapsed.toFixed(0), "ms");
            draggingId = null;
        }

        wasPinching = pinching;
    }

    const hoveredId = computeHoveredId();

    cleanupRemovedNotes();
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

        if (obj.type !== "note" && obj.type !== "chat") {
            drawBox(halfW, halfH, id === hoveredId);
        }

        if (id === selectedId && obj.type !== "note" && obj.type !== "chat") {
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
            const quill = ensureNoteEditor(id);
            const entry = noteEditors[id];
            const pos = worldToScreen(obj.x - obj.size[0] / 2, obj.y - obj.size[1] / 2);

            if (entry.lastX !== obj.x || entry.lastY !== obj.y || entry.lastW !== obj.size[0] || entry.lastH !== obj.size[1]) {
                entry.wrapper.style.left = `${pos.left}px`;
                entry.wrapper.style.top = `${pos.top}px`;
                entry.wrapper.style.width = `${obj.size[0] * getScaleX()}px`;
                entry.wrapper.style.height = `${obj.size[1] * getScaleY()}px`;
                entry.lastX = obj.x;
                entry.lastY = obj.y;
                entry.lastW = obj.size[0];
                entry.lastH = obj.size[1];
            }

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
            const pos = worldToScreen(obj.x - obj.size[0] / 2, obj.y - obj.size[1] / 2);
            entry.wrapper.style.left = `${pos.left}px`;
            entry.wrapper.style.top = `${pos.top}px`;
            entry.sendBtn.style.background = bodyColor;
            entry.sendBtn.style.border = bodyColor;
            entry.wrapper.style.width = `${obj.size[0] * getScaleX()}px`;
            entry.wrapper.style.height = `${obj.size[1] * getScaleY()}px`;

            const textColor = getContrastingTextColor(bgColor);
            Array.from(entry.messagesEl.children).forEach(bubble => {
                if (bubble.className === "typing-indicator") return;
                bubble.style.color = textColor;
                if (bubble.dataset.role === "user") {
                    bubble.style.background = hexToRgba(bodyColor, 0.4);
                }
            });
            
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


    }
    ctx.restore();

    const buttonPos = getButtonPosition();
    const barThickness = 8;

    ctx.fillStyle = buttonColor;
    ctx.shadowColor = buttonColor;
    ctx.shadowBlur = 10;

    ctx.fillRect(
        buttonPos.x,
        buttonPos.y + buttonPos.size / 2 - barThickness / 2,
        buttonPos.size,
        barThickness
    );

    ctx.fillRect(
        buttonPos.x + buttonPos.size / 2 - barThickness / 2,
        buttonPos.y,
        barThickness,
        buttonPos.size
    );

    const removeBtnPos = getRemoveButtonPosition();
    ctx.fillRect(
        removeBtnPos.x,
        removeBtnPos.y + removeBtnPos.size / 2 - barThickness / 2,
        removeBtnPos.size,
        barThickness
    )

    

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

    requestAnimationFrame(draw);
}

function getButtonPosition() {
    return {
        x: canvas.width - 50,
        y: 20,
        size: 40
    }
}

function getRemoveButtonPosition() {
    return {
        x: canvas.width - 50,
        y: canvas.height - 60,
        size: 40
    }
}

canvas.addEventListener("mousedown", (e) => {
    const buttonPos = getButtonPosition();
    if (
        e.offsetX >= buttonPos.x &&
        e.offsetX < buttonPos.x + buttonPos.size &&
        e.offsetY >= buttonPos.y &&
        e.offsetY < buttonPos.y + buttonPos.size
    ) {
        const picker = document.getElementById("widgetPicker");
        picker.style.display = picker.style.display === "none" ? "flex" : "none";
        return;
    }

    const removeBtnPos = getRemoveButtonPosition();
    if (
        e.offsetX >= removeBtnPos.x && e.offsetX < removeBtnPos.x + removeBtnPos.size && e.offsetY >= removeBtnPos.y && e.offsetY < removeBtnPos.y + removeBtnPos.size
    ) {
        if (selectedId !== null) {
            ws.send(JSON.stringify({ type: "remove", obj_id: selectedId }));
            selectedId = null;
        }
        return;
    }

    console.log("mousedown registered");
    const worldX = e.offsetX / getScaleX();
    const worldY = e.offsetY / getScaleY();

    ws.send(JSON.stringify({
        type: "hit_test",
        x: worldX,
        y: worldY
    }));
});

document.addEventListener("mousemove", (e) => {
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
    draggingId = null;
});

canvas.addEventListener("dblclick", (e) => {
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
    panel.style.display = panel.style.display === "none" ? "flex" : "none";
});

document.addEventListener("click", (e) => {
    const panel = document.getElementById("settingsPanel");
    const toggleBtn = document.getElementById("settingsToggle");

    if (panel.style.display === "none") return;

    const clickedInsidePanel = panel.contains(e.target);
    const clickedToggleBtn = toggleBtn.contains(e.target);

    if (!clickedInsidePanel && !clickedToggleBtn) {
        panel.style.display = "none";
    }
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

draw();