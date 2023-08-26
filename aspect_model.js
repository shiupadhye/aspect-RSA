// ------------ priors ------------ 
// verb-object pairs
var verb_objects = {"cleaning":["brush","ocean","street","island"],"solving":["JigsawPuzzle","RubiksCube","crisis","dilemma"],
                   "breaking":["bottle","cage","diamond","OlympicRecord"],
                   "painting":["cabinet","cathedral","city","factory"]}

// duration probabilities
var object_probs = {
  // clean
  'brush': [0.171428571,0.685714286,0.142857143,1e-20,1e-20,1e-20,1e-20,1e-20],
  'island':[1e-20,1e-20,1e-20,0.057142857,0.257142857,0.285714286,0.257142857,0.142857143],
  'ocean':[1e-20,1e-20,1e-20,1e-20,0.028571429,1e-20,0.028571429,0.942857143],
  'street':[1e-20,0.028571429,0.571428571,0.171428571,0.142857143,0.085714286,1e-20,1e-20],
  // paint
  'cabinet':[1e-20,0.2,0.742857143,0.057142857,1e-20,1e-20,1e-20,1e-20],
  'cathedral':[1e-20,1e-20,1e-20,0.171428571,0.228571429,0.371428571,0.171428571,0.057142857],
  'city':[1e-20,1e-20,0.028571429,1e-20,0.057142857,0.142857143,0.371428571,0.4],
  'factory':[1e-20,1e-20,1e-20,0.342857143,0.314285714,0.314285714,0.028571429,1e-20],
  // break
  'bottle':[1,1e-20,1e-20,1e-20,1e-20,1e-20,1e-20,1e-20],
  'cage':[0.085714286,0.485714286,0.228571429,0.114285714,1e-20,0.085714286,1e-20,1e-20],
  'diamond':[0.057142857,0.2,0.257142857,0.228571429,1e-20,0.028571429,0.085714286,0.142857143],
  'OlympicRecord':[0.171428571,0.057142857,0.028571429,1e-20,0.057142857,1e-20,0.628571429,0.057142857],
  // solve
  'JigsawPuzzle':[1e-20,0.171428571,0.571428571,0.228571429,0.028571429,1e-20,1e-20,1e-20],
  'RubiksCube':[0.057142857,0.571428571,0.314285714,0.057142857,1e-20,1e-20,1e-20,1e-20],
  'crisis':[0.028571429,0.142857143,0.257142857,0.257142857,0.114285714,0.057142857,0.114285714,0.028571429],
  'dilemma':[1e-20,0.371428571,0.314285714,0.114285714,0.057142857,0.057142857,0.057142857,0.028571429]
  }


var priorDuration = function(object) {
  var T = [1,60,3600,86400,604800,2419200,31556926,315569260]
  //var T = [2,120,7200,172800,1209600,4838400,63113852,631138520]
  var p = object_probs[object]
  return categorical({ps:p,vs:T})
}

// flat prior over events (1-10)
var priorNEvents = function(){
  return flip(0.5) ? 1 : categorical({vs:_.range(2,11,1)})
}
// prior over event states
var priorEventState = function(object) {
  var D = priorDuration(object)
  var N = priorNEvents()
  return {'duration':D,'numEvents':N}
}

// flat prior on timeframes for temporal adverbials
var priorTimeframe = function(){
  // flat prior on duration for temporal adverbials
  var N = categorical({vs:_.range(1,11,1)})
  // flat prior on time units
  var U = categorical({vs:['second(s)','minute(s)','hour(s)','day(s)','week(s)','month(s)','year(s)']})
  return {'num': N,'unit': U}
}

// generate utterance given verb and object
var phi = 1 - 1e-10
var priorUttr = function(verb,object){
  var timeframe = priorTimeframe()
  return flip(phi) ? {'verb':verb,'object':object,'timeframe': timeframe}: {'verb':verb,'object':object,'timeframe': 'null'}
}

// prior on interpretations
var eventInterpretations = ["episodic","habitual"]
var priorEventInterpretation = function(){
  return uniformDraw(eventInterpretations)
}

// ------------ helper functions ------------ 
// calculate length of activity (in secs) given state
var calcEventSpan = function(state){
  return state.duration * state.numEvents
}
// calculate length of timeframe (in secs) given state
//var calcTimeframe = function(timeframe){
//  var timeUnits = {'second(s)':1,'minute(s)':60,'hour(s)':3600,'day(s)':86400,
//                   'week(s)':604800,'month(s)':2419200,'year(s)':31556926}
//  return timeframe.num * timeUnits[timeframe.unit]
//}

var calcTimeframe = function(timeframe){
  var timeUnits = {'second(s)':2,'minute(s)':120,'hour(s)':7200,'day(s)':172800,
                   'week(s)':1209600,'month(s)':4838400,'year(s)':63113852}
  return timeUnits[timeframe.unit]
}

// generate temporal adverbial string given sampled state
var getTemporalAdverbial = function(timeframe){
  return timeframe.num + " " + timeframe.unit
}
// print utterance
var uttrToString = function(uttr){
  var temporalAdverbial = getTemporalAdverbial(uttr.timeframe)
  return "was" + " " + uttr.verb + " " + "the" + " " + uttr.object + " " + "for" + " " + temporalAdverbial
}

// ------------ semantics ------------ 
var meaning = function(utterance,state,interpretation){
  //print("In meaning: ")
  if (utterance.timeframe != "null") {
    // total span of the activity (based on world state)
    var eventSpan = calcEventSpan(state)
    // reference timeframe in the utterance
    var timeframe = calcTimeframe(utterance.timeframe)
    //print("ES: " + eventSpan)
    //print("TF: " + timeframe)
    // check if event span exceeds timeframe
    var checkEventSpan = timeframe <= eventSpan
    var checkEpisodic = interpretation == "episodic" ? state.numEvents == 1 : state.numEvents > 1
    //print("checkES: " + checkEventSpan)
    //print("checkEpisodic: " + checkEpisodic)
    return checkEventSpan && checkEpisodic
  }
  else {
    return true
  }
}

// ------------ agents ------------ 
// Literal Listener (L0)
var literalListener = function(utterance,interpretation) {
  //print("In L0:")
  return Infer({model: function(){
  var object = utterance.object
  var state = priorEventState(object)
  //var state = {'duration': 300, 'numEvents': 1}
  //print("state (D): " + state.duration + " and state (N): " + state.numEvents)
  var m = meaning(utterance,state,interpretation)
  //print("meaning: " + m)
  condition(m)
  return state
  }})}

// Speaker
// constant cost
var cost = function(utterance){
  return utterance.timeframe == "null"? 2 : 1
}

// set alpha to 1
var alpha = 1
// Speaker (S0)
var speaker = function(verb,object,interpretation,state) {
  return Infer({model: function(){
    var utterance = priorUttr(verb,object)
    var L = literalListener(utterance,interpretation)
    factor(alpha * (L.score(state) - cost(utterance)))
    return utterance
  }})}

// Pragmatic listener (L1)
var pragmaticListener = function(utterance){
  return Infer({method: 'enumerate', 
                strategy: 'likelyFirst',
                model: function(){
    var verb = utterance.verb
    var object = utterance.object
    var interpretation = priorEventInterpretation()
    var state = priorEventState(object)
    observe(speaker(verb,object,interpretation,state),utterance)
    return {'state': state,
            'interpretation': interpretation}
  }})}

var uttr1 = {'verb':"solving",'object':"JigsawPuzzle",'timeframe':{'num':2,'unit':'minute(s)'}}
var d1 = marginalize(pragmaticListener(uttr1),'interpretation')
display(d1)