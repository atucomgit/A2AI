// content.js

// ページロード時に「このサイトを絵文字化」ボタンを表示
const btn = document.createElement("button");
btn.id = "emoji_btn";
btn.innerHTML = "このサイトを絵文字化";
btn.onclick = sendToChatGPT;
btn.style.position = "fixed"; // 絶対パスを指定
btn.style.top = "20px";      // 上からの位置（適宜調整）
btn.style.right = "20px";    // 右からの位置（適宜調整）
btn.style.zIndex = "9999";   // ボタンの表示順（最前面に表示）
btn.style.padding = "10px 20px"; // パディングを追加してボタンを大きくする
btn.style.backgroundColor = "#007BFF"; // ボタンの背景色
btn.style.color = "#FFFFFF"; // ボタンの文字色
btn.style.border = "none"; // ボタンの枠線をなくす
btn.style.borderRadius = "5px"; // ボタンの角を丸くする
btn.style.cursor = "pointer"; // ボタンにカーソルを合わせた時にポインターを表示
document.body.appendChild(btn);

// 「このサイトを絵文字化」ボタンを押すと表示しているWebページの先頭1000文字をChatGPTに送信
function sendToChatGPT() {
  console.log('content.js:sendToChatGPT');
  let text = document.body.innerText.substring(0, 1000);
  sendToChatGPTCore(text);
}

function sendToChatGPTCore(text) {
  console.log('Sending request to ChatGPT...');

  const api_key = 'sk-xxx';
  const endpoint = 'https://api.openai.com/v1/chat/completions';  // エンドポイントの更新

  const data = {
    model: 'gpt-3.5-turbo',
    messages: [
      { role: "system", content: "あなたは文章を絵文字に変換するエキスパートです。絵文字に変換した結果の絵文字だけを簡潔に返信してください。" },
      { role: "user", content: '以下の内容を絵文字のみで表現してください：' + text }
    ],
    temperature: 0.5
  }

  fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + api_key
    },
    body: JSON.stringify(data)
  })
  .then(response => response.json())
  .then(data => {
    console.log('ChatGPT response:', data);

    // ChatGPTの返信を処理する
    if (data.choices && data.choices.length > 0) {
      const emojiContent = data.choices[0]?.message?.content;
      if (emojiContent) {
        showPopup(emojiContent); // ポップアップに表示する関数を呼び出す
      } else {
        console.error('ChatGPT response is invalid.');
      }
    } else {
      console.error('ChatGPT response is invalid.');
    }
  })
  .catch((error) => {
    console.log('Error:', error);
  });
}

function showPopup(content) {
  // ポップアップを表示するコードをここに書く
  // 例えば、以下のようにしてポップアップを表示できます

  const popupDiv = document.createElement('div');
  popupDiv.style.position = 'fixed';
  popupDiv.style.top = '50%';
  popupDiv.style.left = '50%';
  popupDiv.style.transform = 'translate(-50%, -50%)';
  popupDiv.style.padding = '20px';
  popupDiv.style.backgroundColor = '#f0f0f0';
  popupDiv.style.borderRadius = '8px';
  popupDiv.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.2)';
  popupDiv.style.zIndex = '9999';

  const contentDiv = document.createElement('div');
  contentDiv.innerText = content;
  popupDiv.appendChild(contentDiv);

  // クローズボタンを追加
  const closeButton = document.createElement('button');
  closeButton.innerText = 'Close';
  closeButton.style.marginTop = '10px';
  closeButton.style.padding = '8px 16px';
  closeButton.style.backgroundColor = '#007BFF';
  closeButton.style.color = '#ffffff';
  closeButton.style.border = 'none';
  closeButton.style.borderRadius = '4px';
  closeButton.style.cursor = 'pointer';
  closeButton.onclick = () => {
    document.body.removeChild(popupDiv);
  };
  popupDiv.appendChild(closeButton);

  document.body.appendChild(popupDiv);
}
