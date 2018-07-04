<?php
require "utils.php";
control_admin();
?>
<center>
<table border="0" id="extractor_sources" class="extractor_table">
  <tbody>
    <tr><td colspan="5"><h2>Source Management</h2></td></tr>
    <tr class="class0">
      <td>Create a new source:</td>
      <td rowspan="2"><select class="select_source" id="select_source_creation" style="height:60px;"></select></td>
      <td colspan=2> <input alt="" type="text" id="source-url" value="http://" class="extractor_source_url"></td>
      <td rowspan="2" style="width:150px;"><input alt="" type="button" value="create" onclick="add_source()" style="height:60px"></td>
    </tr>
    <tr class="class0">
      <td style="width:150px;"><input alt="" type="text" id="source-name" value="tidzam-"></td>
      <td style="width:150px;"><input alt="" type="button" class="extractor_dateselector" id="dateselector_creation"></td>
      <td><div class="extractor_timeselector" id="timeselector_creation"><span class="time_label" id="time_label_creation"></span></div></td>
    </tr>
    <tr class="class1">
      <td style="width:200px;">Modify a source:</td>
      <td style="width:200px;"><select class="select_source" id="select_source_modify"></select></td>
      <td><input alt="" type="button" class="extractor_dateselector" id="dateselector"></td>
      <td style="width:500px;"><div class="extractor_timeselector" id="timeselector"><span class="time_label" id="time_label"></span></div></td>
      <td><input alt="" type="button" id="source_update" value="update" onclick="modify_source();"></td>
    </tr>
    <tr class="class0">
      <td>Delete a source:</td>
      <td><select class="select_source" id="select_source_deletion"></select></td>
      <td colspan="2"></td>
      <td><input alt="" type="button" id="source_delete" value="remove" onclick="delete_source();"></td>
    </tr>
  </tbody>
</table>

<script>
const livestream_io = io("//tidzam.media.mit.edu/",
      { path: '/livestream/socket.io'});

const analyzer_io = io("//tidzam.media.mit.edu/",
    { path: '/socket.io' ,
     forceNew:true});

var sources           = [];
var databases         = {};
var extraction_rules  = {};
var classifiers       = {};

function req_database_info(){
  analyzer_io.emit('SampleExtractionRules', {"get":"database_info"} );
  analyzer_io.emit('SampleExtractionRules', {"get":"extracted_count"});
  setTimeout(req_database_info, 5000);
}
req_database_info();



function add_source(){
  if ($("#select_source_creation").val() != ""){
    url = null;
    date = $("#dateselector_creation").val().split("/")
    date = date[2] + "-" + date[0] + "-" + date[1] + "-" + $("#time_label_creation").html().replace(/:/g,"-");
  }
  else {
    url  = $("#source-url").val();
    date = null;
  }

  msg = {
        sys:{
          loadsource:{
            name:$("#source-name").val(),
            url:url,
            date:date,
            database:$("#select_source_creation").val(),
            channel:null,
            is_permanent:true
          }
        }
      }
  livestream_io.emit('sys', msg);
  livestream_io.emit('sys', {"sys":{"database":""}});
}

function modify_source(){
  date = $("#dateselector").val().split("/")
  date = date[2] + "-" + date[0] + "-" + date[1] + "-" + $("#time_label").html().replace(/:/g,"-")
  livestream_io.emit("sys", {
    sys:{
      loadsource:{
        name:$("#select_source_modify").val(),
        url:null,
        date:date,
        is_permanent:true
        }
      }
    }
  );
}

function delete_source(){
    msg =  {sys:{unloadsource:{name:$("#select_source_deletion").val()} } }
    livestream_io.emit("sys", msg);
}

function sources_add(source){
  sources.push(source);
  // Add a line for stream control
  for (var i=0;i<sources.length;i++){
    s = sources[i].split("-out")[0]
    if (sources.indexOf(s) == -1)
      sources_add(s)
  }
  // Add lines for the control each channel of the stream
  cl = "class" + ($("#extractor_extractor tbody").children().length % 2);
  list = '<tr class="'+cl+'">';
  list += '<td class="extractor_extractor_time"><span id="time_'+source+'"></span></td>';
  list += '<td class="extractor_extractor_name">'+source+'</td>';
  list += '<td class="extractor_extractor_result"><span id="result_'+source+'"></span></td>';
  list += '<td class="extractor_extractor_extract"><select class="extract_list" id="extract_'+source+'" multiple>';
  list += '<option></option>'
  list += '<option>unknown</option>'
  for (var j=0; j<classifiers.length;j++)
    list += '<option>' + classifiers[j] + '</option>'
  list += '</select></td>';
  list +='<td><span class="extract_list_span" id="extract_list_span_'+source+'"></td>';
  list += '<td><input type="text" id="extract_length_'+source+'" value="0.5"></td>'
  list += '<td><select id="extract_rate_'+source+'"><option label="">auto</option>';
  for (var j=1; j>=0; j -= 0.1)
    list += '<option>'+ (Math.round(j*100)/100) +'</option>';
  list += '</select></td>';
  list += '<td><span id="extract_count_'+source+'"></span></td>'
  list += '<td><input alt="" type="image" src="static/img/play.png" class="extractor_btn_play" id="source_btn_'+source+'"></td>';
  list += '</tr>';
  $("#extractor_extractor tbody").append(list);

  $("#source_btn_"+source).click(function(){
    $( '#audio_player'  ).attr('src', "http://tidzam.media.mit.edu/audio/"+this.id.split("source_btn_")[1]+".ogg");
    $( '#audio_player'  ).load();
    $( '#audio_player' ).trigger('play');
  })

}

function extractor_extractor_update(data){
  databases = data
  var list = '<option label=""></option>'
  for (var database in databases)
    list += '<option label="">'+database+'</option>';
  $('.select_source').each(function(){
    $(this).html(list);
  })
}

function sources_datepicker_update(database, divID){
  var eventDates = {}
  for (var i=0;i < database["database"].length; i++){
    f = database["database"][i]
    start = f[0].split("-");
    start_date = new Date(start[0],start[1]-1,start[2])
    end   = f[1].split("-");
    end_date = new Date(end[0],end[1]-1,end[2])
    for (var d = start_date; d <= end_date; d.setDate(d.getDate() + 1))
      eventDates[d] = new Date(d).toString();
    }
    $( "#" + divID).datepicker("destroy");
    $( "#" + divID).datepicker({
     beforeShowDay: function(date) {
       if (eventDates[date]) return [true, "event", eventDates[date]];
       else                  return [true, '', ''];
       }
     });
  }

$('#select_source_modify').change(function(){
  sources_datepicker_update(databases[$("#select_source_modify").val()], "dateselector");
});

$('#select_source_creation').change(function(){
  if ($("#select_source_creation").val() != ""){
    $("#source-url").prop("disabled", true);
    $("#dateselector_creation").prop("disabled", false);
    $("#timeselector_creation").prop("disabled", false);

  }
  else {
      $("#source-url").prop("disabled", false);
      $("#dateselector_creation").prop("disabled", true);
      $("#timeselector_creation").prop("disabled", true);
    }

  sources_datepicker_update(databases[$("#select_source_creation").val()], "dateselector_creation");
});

$( "#timeselector" ).slider({range: false,
  max: 24 * 60 * 60,
  slide: function(event, ui) {
    database = this.id.split("_");
    database = database[database.length-1];
    var hours   = Math.floor(ui.value / 3600);
    var minutes = Math.floor((ui.value - (hours * 3600)) / 60);
    var seconds = ui.value - (hours * 3600) - (minutes * 60);
    $( "#time_label" ).text(hours+":"+minutes+":"+seconds)
  }
});

$( "#timeselector_creation" ).slider({range: false,
  max: 24 * 60 * 60,
  slide: function(event, ui) {
    database = this.id.split("_");
    database = database[database.length-1];
    var hours   = Math.floor(ui.value / 3600);
    var minutes = Math.floor((ui.value - (hours * 3600)) / 60);
    var seconds = ui.value - (hours * 3600) - (minutes * 60);
    $( "#time_label_creation" ).text(hours+":"+minutes+":"+seconds)
  }
});

ready = false
livestream_io.on('sys', function(data){
  if (data.sys)
    if(data.sys.database){
      ready = true
      extractor_extractor_update(data.sys.database);
    }
  });

function load_sources_list(){
  console.log("load " + ready)
  if(ready) return
  livestream_io.emit('sys', {"sys":{"database":""}});
  }
  setTimeout(load_sources_list,500)

$(document).ready(function(){
  load_sources_list()
});

</script>
