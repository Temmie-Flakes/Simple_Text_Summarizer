print("importing dependiencies")
import gradio as gr
import argparse
import torch
import sys
import base64
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


def parse_args():
    parser = argparse.ArgumentParser(description='A Simple Gradio implemation for summary models')
    parser.add_argument('--model', default='pszemraj/led-base-book-summary',
                        help='Hugging Face directory of the model to use. format: (userName/modelName)')
    return parser.parse_args()
    
args = parse_args()
#dont know if i should use with torch.no_grad(): or not
print("loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.model,cache_dir="modelsCache/"+args.model.split("/")[1])
print("loading model")
model = AutoModelForSeq2SeqLM.from_pretrained(args.model,cache_dir="modelsCache/"+args.model.split("/")[1])

print("building summarizer pipeline")
summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
)
#the different possable arguements for the loaded model. tp in this case
modelArgs=[
"max_length",
"min_length",
"temperature",
"num_beams",
"top_k",
"top_p",
"typical_p",
"num_return_sequences",
"num_beam_groups",
"diversity_penalty",
"no_repeat_ngram_size",
"encoder_no_repeat_ngram_size",
"repetition_penalty",
"length_penalty",
"early_stopping",
"renormalize_logits",
"do_sample",
"use_cache",
"remove_invalid_values",
"synced_gpus"
]
#get argument default data type (model.generate.__wrapped__.__annotations__["early_stopping"].__args__[0])
#(model.generate.__wrapped__.__annotations__[modelArgs[i]].__args__[0])
#model.generation_config.__dict__
#(type(model.generation_config.__dict__))
#if((type(model.config.__dict__[(list(model.config.__dict__.keys()))[i]])))==int:
webUiInteractions=[]
defaultModelArgsValues=[]
validModelKwargs=[]
argumentDict={}

#assigns the gradio interface type for each argument to be able to control them
#checks python version
if sys.version_info[1] == 10:
    #Need to impliment case where continue is called when model.generate.__wrapped__.__annotations__[modelArgs[i]] dosent exist
    for i in range(len(modelArgs)):
        try:
            defaultArg=model.config.__dict__[modelArgs[i]]
        except:
            defaultArg=None
        defaultModelArgsValues.append(defaultArg)
        validModelKwargs.append(modelArgs[i])
    defaultModelArgsValues[validModelKwargs.index("use_cache")]=True
    for i in range(len(validModelKwargs)):
        #yes i know i could have done a switch statement. suck my ween
        if(model.generate.__wrapped__.__annotations__[validModelKwargs[i]].__args__[0])==int:
            webUiInteractions.append(gr.Number(label=(validModelKwargs[i]),value=defaultModelArgsValues[i],precision=0))
        if(model.generate.__wrapped__.__annotations__[validModelKwargs[i]].__args__[0])==float:
            webUiInteractions.append(gr.Number(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))
        if(model.generate.__wrapped__.__annotations__[validModelKwargs[i]].__args__[0])==str:
            webUiInteractions.append(gr.Textbox(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))
        if(model.generate.__wrapped__.__annotations__[validModelKwargs[i]].__args__[0])==bool:
            webUiInteractions.append(gr.Checkbox(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))
else:
    for i in range(len(modelArgs)):
        try:
            defaultArg=model.config.__dict__[modelArgs[i]]
        except:
            try:
                defaultArg=model.generation_config.__dict__[modelArgs[i]]
            except:
                continue
                #dont ask
        defaultModelArgsValues.append(defaultArg)
        validModelKwargs.append(modelArgs[i])
    defaultModelArgsValues[validModelKwargs.index("use_cache")]=True
    for i in range(len(validModelKwargs)):
        if(type(model.generation_config.__dict__[validModelKwargs[i]]))==int:
            webUiInteractions.append(gr.Number(label=(validModelKwargs[i]),value=defaultModelArgsValues[i],precision=0))
        if(type(model.generation_config.__dict__[validModelKwargs[i]]))==float:
            webUiInteractions.append(gr.Number(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))
        if(type(model.generation_config.__dict__[validModelKwargs[i]]))==str:
            webUiInteractions.append(gr.Textbox(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))
        if(type(model.generation_config.__dict__[validModelKwargs[i]]))==bool:
            webUiInteractions.append(gr.Checkbox(label=(validModelKwargs[i]),value=defaultModelArgsValues[i]))

#combines the argument names (modelArgs) and the default values from the loaded model into a new dict

def largeInputPipeline(batch, newArgs):
    print("using larger batch processing and attention masks")
    print("preparing input...")
    #with torch.no_grad(): # everything except result
    inputs_dict = tokenizer(batch, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
    input_ids = inputs_dict.input_ids.to("cuda")
    attention_mask = inputs_dict.attention_mask.to("cuda")
    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    print("starting summarization...")
    predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, **newArgs)
    print("preparing output...")
    batchResult = tokenizer.batch_decode(predicted_abstract_ids, skip_special_tokens=True)
    return batchResult
    
for A, B in zip(validModelKwargs, defaultModelArgsValues):
    argumentDict[A] = B
    
#main handler function
def generateOutput(wall_of_text,txtFile,isLargeText, *newArgs):
    for i in range(len(validModelKwargs)):
         argumentDict[validModelKwargs[i]] = newArgs[i]
         
    if txtFile:
        #aparently gradio file system dosent retain origional binary information and converts \n into \r\n
        #also gradio forces temp files for "security" and using binary mode is not supposed to do that acording to the docs 
        #with default settings it makes acually makes 2 identical temp files. which is stupid. atleast i cut it in half with this method -_-
        text=txtFile.replace(b'\r\n',b'\n').decode()
        print("|using file|")
    else:
        text=wall_of_text
    
    if(isLargeText):
        result=largeInputPipeline(text,argumentDict)
        print(result)
        result=result[0]
    else:
        print("starting summarization...")
        #with torch.no_grad():# just result line
        result=summarizer(text,**argumentDict)
        print(result)
        result=result[0]["summary_text"]
    
    return (result.__str__()), len(tokenizer.tokenize(str(text)))+2, len(tokenizer.tokenize(str(result)))+2

#
inputTokens = gr.Number(label="input token count")
outputTokens = gr.Number(label="output token count")
file=gr.File(type="binary")
isBigFile=gr.Checkbox(label="check me if large text ~>10,000 words",value=False)
demo = gr.Interface(fn=generateOutput, inputs=[gr.Textbox(lines=30),file,isBigFile,*webUiInteractions], outputs=[gr.Textbox(),inputTokens,outputTokens])

demo.launch(inbrowser=True,inline=True)   