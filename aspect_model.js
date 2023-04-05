// ------------ priors ------------ 
// verb-object pairs
var verb_objects = {"cleaning":["table","ocean"]}

// duration probabilities
var object_probs = {'table':[0.05,0.05,0.2,0.3,0.2,0.1,0.05,0.05],
                 'ocean':[0.05,0.05,0.05,0.05,0.05,0.05,0.2,0.5]}

// prior over durations
var priorDuration = function(object) {
  var T = [1,60,3600,86400,604800,2630000,31600000,316000000]
  var p = object_probs[object]
  return categorical({ps:p,vs:T})
}

// flat prior over events (1-10)
var priorNEvents = function() {
 return categorical({vs:_.range(1,11,1)})
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
var phi = 0.9
var priorUttr = function(verb,object){
  var timeframe = priorTimeframe()
  return flip(0.9) ? {'verb':verb,'object':object,'timeframe': timeframe}: {'verb':verb,'object':object,'timeframe': 'null'}
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
var calcTimeframe = function(timeframe){
  var timeUnits = {'second(s)':1,'minute(s)':60,'hour(s)':3600,'day(s)':86400,'week(s)':604800,
                   'month(s)':2630000,'year(s)':31600000}
  return timeframe.num * timeUnits[timeframe.unit]
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
  return utterance.timeframe == "null"? 1 : 2
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

//var uttr = {'verb':"cleaning",'object':"ocean",'timeframe':{'num':1,'unit':'second(s)'}}
var nullUttr = {'verb':"cleaning",'object':"ocean",'timeframe':"null"}
marginalize(pragmaticListener(nullUttr),'interpretation')