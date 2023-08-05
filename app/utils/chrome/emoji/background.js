chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action == "convert") {
      fetch('https://api.openai.com/v1/engines/davinci-codex/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer API KEY'
        },
        body: JSON.stringify({
          prompt: request.text.slice(0,1000),
          temperature: 0.5,
          max_tokens: 60
        })
      }).then(response => response.json())
      .then(data => sendResponse({result: data.choices[0].text}))
      .catch(error => console.error('Error:', error));
    }
    return true;  // Will respond asynchronously.
  });