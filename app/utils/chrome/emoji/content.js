// content.js

// ページロード時に「このサイトを絵文字化」ボタンを表示
let btn = document.createElement("button");
btn.id = "emoji_btn";
btn.innerHTML = "このサイトを絵文字化";
btn.onclick = sendToChatGPT;
document.body.insertBefore(btn, document.body.childNodes[0]);

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
      { role: "system", content: "あなたは文章を絵文字に変換するエキスパートです。絵文字に変換した結果の絵文字だけを完結に返信してください。" },
      { role: "user", content: '以下の内容を絵文字に変換してください：' + text }
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
  popupDiv.style.top = '10px';
  popupDiv.style.left = '50%';
  popupDiv.style.transform = 'translateX(-50%)';
  popupDiv.style.padding = '10px';
  popupDiv.style.backgroundColor = '#ffffff';
  popupDiv.style.border = '1px solid #000000';
  popupDiv.style.zIndex = '9999';
  popupDiv.innerText = content;

  document.body.appendChild(popupDiv);
}
