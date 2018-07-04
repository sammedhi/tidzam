<?php
require "utils.php";
control_admin();
?>
<center>
<table border="0" id="extractor_extractor" class="extractor_table">
  <tbody>
    <tr><td colspan="7"><h2>Sample Database</h2></td></tr>
      <tr>
          <td colspan="3" style="width:50%">
            <div id="database_info_div_primary" class="database_info_div"></div>
          </td>
          <td colspan="4" style="width:50%">
            <div id="database_info_div_birds" class="database_info_div"></div>
          </td>
      </tr>
    <tr><td colspan="7"><h2>Automatic Sample Extraction Configuration</h2></td></tr>
    <tr class="extractor_table_titles">
      <td style="width:250px;">Datetime</td>
      <td style="width:250px;">Source</td>
      <td style="width:250px;">Analyze</td>
      <td style="min-width:270px;">Extraction Classes</td>
      <td>Extraction Classes</td>
      <td style="min-width:270px;">Length (seconds)</td>
      <td style="min-width:270px;">Object Fitler</td>
      <td style="width:50px;">Rate</td>
      <td style="width:50px;">Count</td>
      <td style="width:50px;"><input alt="" type="button" value="Apply Rules" onclick="send_extract_configuration();"></td>
    </tr>
  </tbody>
</table>
</center>
<audio controls id="audio_player" style="opacity:0;">
  <source src="" type="audio/wav">
</audio>

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

function send_extract_configuration(){
  analyzer_io.emit('SampleExtractionRules', {"set":"rules", "rules":extraction_rules} );
  console.log("ICI " + JSON.stringify({"set":extraction_rules}))
}


function extraction_rules_update(source){
  try {
    $("#extract_list_span_"+source).html($("#extract_" +source).val().join());
    extraction_rules[source] = {
      classes:$("#extract_" +source).val().join(),
      rate:$("#extract_rate_" +source).val(),
      length:$("#extract_length_"+source).val(),
      object_filter:$("#extract_object_filter_"+source).is(':checked')
      }

    if (source.split("-out").length == 1){
      channels = $(".extract_list_span")
      for (var i=0; i<channels.length; i++)
        if (channels[i].id.indexOf(source) != -1){
          $("#"+channels[i].id).html($("#extract_" +source).val().join());
          ch = channels[i].id.split("extract_list_span_")[1];
          extraction_rules[ch] = {
            classes:$("#extract_" +source).val().join(),
            rate:$("#extract_rate_" +source).val(),
            length:$("#extract_length_"+source).val(),
            object_filter:$("#extract_object_filter_"+source).is(':checked')
            }
          }
        }
      }
    catch(err){
        console.log("No classe selected when rate change")
    }
  }

var database_info_chart_primary = new google.visualization.PieChart(document.getElementById('database_info_div_primary'));
var database_info_chart_birds   = new google.visualization.PieChart(document.getElementById('database_info_div_birds'));
analyzer_io.on('SampleExtractionRules',function(obj){
  //console.log(JSON.stringify(obj));
  // Receive extraction rules
  if (obj.rules && obj.rules != ""){
    for (var source in obj.rules){
      $("#extract_list_span_" +source).text(obj.rules[source].classes);
      $("#extract_rate_" +source).val(obj.rules[source].rate);
    }
  }

  if(obj.extracted_count){
    for (var key in obj.extracted_count){
      $("#extract_count_" +key).html( obj.extracted_count[key]);
    }
  }

  // Receive sample database information
  if(obj.database_info){
    var data_primary = [ ["Classe","Size"] ];
    var data_birds   = [ ["Classe","Size"] ];

    for (cl in obj.database_info){
      if (cl.indexOf("birds-") == -1)
            data_primary.push([cl, obj.database_info[cl]]);
      else  data_birds.push([cl, obj.database_info[cl]]);
    }

    data_primary = google.visualization.arrayToDataTable(data_primary);
    data_birds   = google.visualization.arrayToDataTable(data_birds);

    setTimeout(function(){
      database_info_chart_primary.draw(data_primary, {title: 'Sample Database (Primary Classes)'});
      database_info_chart_birds.draw(data_birds, {title: 'Sample Database (Birds Classes)'});
    },500);
  }
});

analyzer_io.on('sys',function(data){
  //console.log(data)
  if (data.sys){
    if (data.sys.classifier){
      if (data.sys.classifier.list)
        classifiers = data.sys.classifier.list;
      }
    }

  else for (var i=0; i < data.length; i ++){
      channel = data[i].chan.replace(":","-");
      if ( sources.indexOf(channel) == -1 ){
        sources_add(channel);
        analyzer_io.emit('SampleExtractionRules', {"get":"rules" } );

        setTimeout( function(){
          livestream_io.emit('sys', {"sys":{"database":""}});
        },500);
      }

      $("#time_"+channel).html(data[i].analysis.time);
      $("#result_"+channel).text(data[i].analysis.result);
    }
  });
analyzer_io.emit('sys', {sys:{classifier: {list:''}}} );

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
  list += '<td><input id="extract_object_filter_'+source+'" type="checkbox"></td>'
  list += '<td><select id="extract_rate_'+source+'"><option label="">auto</option>';
  for (var j=1; j>=0; j -= 0.05)
    list += '<option>'+ (Math.round(j*100)/100) +'</option>';
  list += '</select></td>';
  list += '<td><span id="extract_count_'+source+'"></span></td>'
  list += '<td><input alt="" type="image" src="static/img/play.png" class="extractor_btn_play" id="source_btn_'+source+'"></td>';
  list += '</tr>';
  $("#extractor_extractor tbody").append(list);

  $("#extract_"+ source).change(function(){
    console.log("PLOP")
    extraction_rules_update(this.id.split("extract_")[1])
  });

  $("#extract_rate_"+ source).change(function(){
    console.log("PLOP rate")
    extraction_rules_update(this.id.split("extract_rate_")[1])
  });

  $("#extract_length_"+ source).change(function(){
    console.log("PLOP length")
    extraction_rules_update(this.id.split("extract_length_")[1])
  });

  $("#extract_object_filter_"+ source).change(function(){
    console.log("PLOP object filter")
    extraction_rules_update(this.id.split("extract_object_filter_")[1])
  });



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

livestream_io.on('sys', function(data){
  if (data.sys)
    if(data.sys.database)
      extractor_extractor_update(data.sys.database);
  });

$(document).ready(function(){
  ;
})

</script>
