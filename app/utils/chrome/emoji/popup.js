// popup.js
// background.jsからのメッセージを受け取って結果を表示する
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.result) {
    document.getElementById('emoji_output').innerText = request.result;
  }
});
