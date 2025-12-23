let previousTimestamp = null;  
const ws = new WebSocket("ws://0.0.0.0:8001/ws/fdfb8545-c177-48a2-bdce-b06af2032092_test_poc");
  
ws.onmessage = function(event) {  
    const messages = document.getElementById('messages');  
    const data = event.data;
    const jsonStr = data.substring(data.indexOf("JSONSTART") + 9, data.indexOf("JSONEND"));
    const messageData = JSON.parse(jsonStr);
    
    console.log('Received message:', messageData);
    const currentTimestamp = messageData.timestamp;  
    const showTimestamp = previousTimestamp  
        ? (currentTimestamp - previousTimestamp >= 300)  
        : true;  
  
    if (showTimestamp && currentTimestamp) {
        previousTimestamp = currentTimestamp;  
        const timestampDiv = document.createElement('div');  
        timestampDiv.classList.add('timestamp');  
        timestampDiv.textContent = formatTime(currentTimestamp);  
        messages.appendChild(timestampDiv);  
    }  
  
    const messageWrapper = document.createElement('div');  
    messageWrapper.classList.add('message-wrapper');  
  
    const role = document.createElement('div');  
    role.classList.add('role');  
    role.textContent = messageData.role;  
  
    const icon = document.createElement('div');  
    icon.classList.add('icon');  
    icon.textContent = getInitials(messageData.role);  
    icon.style.background = getGradient(messageData.role);  
  
    const message = document.createElement('li');  
    message.classList.add('server-message');  
    const md = window.markdownit();  
    message.innerHTML = md.render(messageData.content);  
    messageWrapper.appendChild(icon);  
    messageWrapper.appendChild(role);  
    messageWrapper.appendChild(message);  
    messages.appendChild(messageWrapper);  
    scrollToBottom();  
};
  
function sendMessage(event) {  
    const input = document.getElementById("messageText");  
    const messages = document.getElementById('messages');  
  
    const messageWrapper = document.createElement('div');  
    messageWrapper.classList.add('message-wrapper', 'user-message-wrapper');  
  
    const userMessage = document.createElement('li');  
    userMessage.classList.add('user-message');  
    const userContent = document.createTextNode(input.value);  
    userMessage.appendChild(userContent);  
  
    const currentTimestamp = Math.floor(Date.now() / 1000);  
    const showTimestamp = previousTimestamp  
        ? (currentTimestamp - previousTimestamp >= 300)  
        : true;  
  
    if (showTimestamp) {  
        previousTimestamp = currentTimestamp;  
        const timestampDiv = document.createElement('div');  
        timestampDiv.classList.add('timestamp');  
        timestampDiv.textContent = formatTime(currentTimestamp);  
        messages.appendChild(timestampDiv);  
    }  
  
    messageWrapper.appendChild(userMessage);  
    messages.appendChild(messageWrapper);  
  
    ws.send(JSON.stringify({content: input.value, timestamp: currentTimestamp}));  
    input.value = '';  
    event.preventDefault();  
    scrollToBottom();  
}  
  
function formatTime(timestamp) {  
    const date = new Date(timestamp * 1000);  
    return date.toLocaleTimeString();  
}  
  
function getInitials(role) {  
    const words = role.split(' ');  
    const initials = words.map(word => word[0].toUpperCase()).join('');  
    return initials.slice(0, 2); // Take at most two initials  
}  
  
function getGradient(initials) {  
    const hashCode = s => s.split('').reduce((a, b) => {  
        a = ((a << 5) - a) + b.charCodeAt(0);  
        return a & a;  
    }, 0);  
  
    const intToRGB = i => {  
        const c = (i & 0x00FFFFFF)  
            .toString(16)  
            .toUpperCase();  
        return "00000".substring(0, 6 - c.length) + c;  
    };  
  
    const color1 = intToRGB(hashCode(initials + '1'));  
    const color2 = intToRGB(hashCode(initials + '2'));  
    return `linear-gradient(135deg, #${color1}, #${color2})`;  
}  
  
function scrollToBottom() {  
    const messagesContainer = document.getElementById('messages-container');  
    messagesContainer.scrollTop = messagesContainer.scrollHeight;  
}  



