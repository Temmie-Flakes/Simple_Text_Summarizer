A simple implementation and UI of Summary Transformer models. Specifically The pszemraj/led-large-book-summary

am new to git and still learning. i have no clue what im doing. its like moving through quicksand.

quick install (windows):<br />
----
1. download this repo as a zip or using Git.
2. download and install python 3.10.6 (recommended) OR latest python version (tested on 3.11.3) and add to PATH
3. run Install.bat
4. run either RunBaseModel.bat or RunLargeModel.bat depending if you want the bigger model version.

<br /><br />
how to use:
----
- put large amounts of text in the text box -[OR]-drag and drop text files 
    - (if there is a text file inserted it will disregard anything in the text box.)
- scroll to the bottom and hit generate.
<br />

all the files get stored locally and can be used offline once ran atleast one time.<br />
all nessarry files \*__should__\* all get stored within the same file as the program

yes i know code is jank. ill polish it later
<br />

theoretically you can import summary and text-to-text models by passing the python file with the name of the huggingface repo <br />
you can edit the `venv\Scripts\python.exe Open_WebUI.py [hugging face repo name]` line in the batch file. look at `RunLargeModel.bat` for example.<br />
*this feature is untested*


Todo:
----
- [ ] use Gradio Blocks insted of default gradio interface
- [ ] force gradio to stop making temporary files when arleady using a text file
- [ ] add ability to change summary models while open
- [ ] add ability to save input and output as text files
- [ ] add support for other models. (some models may arleady work... untested)
- [ ] look into merging as an extension with oobabooga text generator
