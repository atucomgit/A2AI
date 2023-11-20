function doGet(e) {
  var calendarId = 'primary'; // これで実行者のカレンダーIDが取得できる
  var calendar = CalendarApp.getCalendarById(calendarId);
  
  var today = new Date(); // 今日の日付を取得します
  var startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1); // 当月の初日を取得します
  var endOfMonth = new Date(today.getFullYear(), today.getMonth() + 3, 0); // 3か月後の末日を取得します
  
  var events = calendar.getEvents(startOfMonth, endOfMonth);
  
  var jsonContent = []; // JSONの配列を作成します
  
  for (var i = 0; i < events.length; i++) {
    var event = events[i];
    var title = event.getTitle();
    var startTime = event.getStartTime();
    var endTime = event.getEndTime();
    var location = event.getLocation();
    var description = event.getDescription().replace(/(\r\n|\n|\r)/gm, ' '); // 改行コードをスペースに置換します
    var attendees = event.getGuestList().map(function(guest) {
      return guest.getEmail();
    });
    
    var year = startTime.getFullYear();
    var month = startTime.getMonth() + 1;
    var day = startTime.getDate();
    var dayOfWeek = getDayOfWeek(startTime);
    var startHour = ('0' + startTime.getHours()).slice(-2);
    var startMinute = ('0' + startTime.getMinutes()).slice(-2);
    var endHour = ('0' + endTime.getHours()).slice(-2);
    var endMinute = ('0' + endTime.getMinutes()).slice(-2);
    
    var eventObj = {
      '年': year,
      '月': month,
      '日': day,
      '曜日': dayOfWeek,
      '開始時刻': startHour + ':' + startMinute,
      '終了時刻': endHour + ':' + endMinute,
      'イベント名': title,
      '場所': location,
      '説明': description,
      '参加者': attendees
    };
    
    jsonContent.push(eventObj);
  }
  
  var output = ContentService.createTextOutput(JSON.stringify(jsonContent));
  output.setMimeType(ContentService.MimeType.JSON);
  
  return output;
}

function getDayOfWeek(date) {
  var days = ['日', '月', '火', '水', '木', '金', '土'];
  return days[date.getDay()];
}