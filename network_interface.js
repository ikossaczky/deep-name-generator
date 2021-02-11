// get HTML elements
const wordStartElem = document.getElementById("word_start");
const randomnessElem = document.getElementById("randomness");
const modelElem = document.getElementById("model")
const generateElem = document.getElementById("generate")
const outputElem = document.getElementById("output")
const networkPictureElem = document.getElementById("network_picture")

// initialization
var model_path = "https://raw.githubusercontent.com/ikossaczky/deep-name-generator/master/models/"
var models = {active: null};
var coder_specs = {name: "coder_specs"};
registered_model_names=[
    "english-names", "english-places",
    "german-names", "german-places",
    "slovak-names", "slovak-places"];
for (model_name of registered_model_names){
    document.getElementById('model').add(new Option(model_name, model_name))
}
modelElem.addEventListener("change", loadModel)
generateElem.addEventListener("click", predict)
x=loadModel(modelElem.value)

// function for model loading
async function loadModel(){
    model_name = modelElem.value
    // set the model to be active
    models.active = model_name
    // if model was not loaded yet, load it:
    if (!(model_name in models)){
        generateElem.disabled = true
        // loading encoder-decoder specs
        console.log("loading coder_specs for model "+model_name)
        $.getJSON(model_path+model_name+"/coder.json",
         function(data){
            coder_specs[model_name]=data;
            console.log("successfully loaded coder_specs for model "+model_name)
        })
        // loading tensorflow model
        console.log("loading model "+model_name)
        var loaded_model = await tf.loadLayersModel(model_path+model_name+"/model.json")
        models[model_name] = loaded_model
        generateElem.disabled = false
    }
    else {
        generateElem.disabled = false
    }
    networkPictureElem.src=model_path+model_name+"/model.png"
    //networkPictureElem.style="width: 50%; height: 50%;"
}
// single character prediction function
async function predict(){
    var randomness = randomnessElem.value;
    var new_character;
    var token_wordstart = coder_specs[models.active].start_token + wordStartElem.value
    for (var k=0; k<coder_specs[models.active].max_word_size - wordStartElem.value.length; k++){
        var input_tensor = tf.tensor(encode(token_wordstart));
        var output_tensor = await models[models.active].predict(input_tensor);
        var output_ar = await output_tensor.array();
        new_character = decodeCharacter(output_ar[0][token_wordstart.length-1], randomness);
        if (new_character==coder_specs[models.active].end_token){
            break;
        }
        else{
            token_wordstart=token_wordstart+new_character
        }
    }
    console.log(token_wordstart.substring(1))
    outputElem.innerHTML = token_wordstart[1].toUpperCase() + token_wordstart.substring(2)
}
// word to one-hot encoding function
function encode(word){
    const max_word_size = coder_specs[models.active].max_word_size
    const onehot_size = coder_specs[models.active].onehot_size
    const chardict = coder_specs[models.active].chardict
    var word = word.toLowerCase() + coder_specs[models.active].end_token.repeat(max_word_size-word.length)
    var code = [[]]
    for (var k=0; k<max_word_size; k++){
        code[0][k]=new Array(onehot_size).fill(0.0)
        code[0][k][chardict[word[k]]]=1.0
    }
    return code
}
// probability to character decoding function
function decodeCharacter(probabilities, randomness){
    if (randomness>0.0){
        var rescaled = probabilities.map( (x)=> {return x**(1/randomness)})
        var norm_factor = rescaled.reduce((a, b) => a + b, 0)
        var new_probs = rescaled.map((x)=>x/norm_factor)
        var charnum = randomIntegerWithProbs(new_probs)
    }
    else{
        var charnum = argMax(probabilities) 
    }
    return coder_specs[models.active].charlist[charnum]
}
// helper function for probability sampling
function randomIntegerWithProbs(probabilities){
    var cummulative_prob=0.0
    var num = 0
    var sampled_prob = Math.random()
    for (var prob of probabilities){
        cummulative_prob = cummulative_prob + prob
        if (sampled_prob <= cummulative_prob){
            return num
        }
        else {
            num++
        }
    }
    return probabilities.length -1 // just to be safe in case of rounding problems
}
// helper argmax function from https://gist.github.com/engelen/fbce4476c9e68c52ff7e5c2da5c24a28
function argMax(array) {
  return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}