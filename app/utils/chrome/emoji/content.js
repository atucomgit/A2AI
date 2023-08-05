chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action == "convert") {
      let text = document.body.innerText;
      chrome.runtime.sendMessage({action: "convert", text: text}, (response) => {
        alert(response.result);
      });
    }
  });