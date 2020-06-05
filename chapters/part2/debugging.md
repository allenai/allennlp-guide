---
title: 'Debugging your AllenNLP code'
description: "Some tips and tricks for using an IDE debugger with AllenNLP."
type: chapter
---

<textblock>

This is a guide to debugging AllenNLP code in both PyCharm and VSCode.  If you have a different
preferred debugger, these instructions will hopefully get you most of the way there.

</textblock>


<exercise id="1" title="Some Code to Debug">

We'll demonstrate the debugger using the Academic Paper Classifier model from our
["AllenNLP-as-a-Library" example](https://github.com/allenai/allennlp-as-a-library-example).

If you'd like to follow along, clone that repo and install its requirements.  Note that it uses an
older version of AllenNLP, but the principles here still apply (we'll update this chapter to use
newer source material eventually).

</exercise>


<exercise id="2" title="How to Debug in PyCharm, Using 'Run > Attach to Local Process'">

Our recommended workflow is using our command-line tool `allennlp`.
The example repo contains the training command:

```
allennlp train experiments/venue_classifier.json -s /tmp/your_output_dir_here --include-package my_library -o '{"trainer": {"cuda_device": -1}}'
```

(I added an override to train on the CPU, since the machine you're running PyCharm on probably doesn't have a GPU.)

After which you can select "Run > Attach to Local Process",

<img src="/part2/debugging/attach_to_process_1.png" width=250em />

search for the one that's running `allennlp`,

<img src="/part2/debugging/attach_to_local_process.png" width=650em />

and get results in the debugger:

<img src="/part2/debugging/attach_to_process_3.png" width=650em />

</exercise>


<exercise id="3" title="How to Debug in PyCharm, Using 'Run > Debug'">

Some people find the "attach to local process" workflow to be a nuisance,
especially if you are repeatedly debugging things.

Fortunately, it's easy to launch the training process from code,
using `allennlp.commands.main()`.

First, create a wrapper script that puts your command-line args into `sys.argv` and calls `main()`. In the academic paper classifier example, I could use the following:

```python
import json
import shutil
import sys

from allennlp.commands import main

config_file = "experiments/venue_classifier.json"

# Use overrides to train on CPU.
overrides = json.dumps({"trainer": {"cuda_device": -1}})

serialization_dir = "/tmp/debugger_train"

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "my_library",
    "-o", overrides,
]

main()
```

Then you can set a breakpoint somewhere in your code:

<img src="/part2/debugging/breakpoint.png" width=400em />

Go back to the wrapper script, right-click in the editor and choose "Debug":

<img src="/part2/debugging/debug_debug.png" width=300em />

And then once you hit your breakpoint, you can inspect variables or step through code to your heart's content:

<img src="/part2/debugging/inspect_variable.png" width=550em />

</exercise>



<exercise id="4" title="How to Debug in VSCode">

You can use the same debug wrapper script as in the previous section.

Then click on the debug icon all the way on the left,

<img src="/part2/debugging/vscode_debug_icon.png" width=50em />

select "Python > Current File" as the debugging profile,

<img src="/part2/debugging/vscode_debugging_profile.png" width=300em />

set a breakpoint somewhere,

<img src="/part2/debugging/vscode_breakpoint.png" width=300em />

go back to the wrapper script, and select "Debug > Start Debugging" from the menu.

<img src="/part2/debugging/vscode_start_debugging.png" width=300em />

Your code will run in the integrated terminal, and when you hit
your breakpoint you'll be able to inspect variables

<img src="/part2/debugging/vscode_watched_variable.png" width=500em />

and the call stack

<img src="/part2/debugging/vscode_call_stack.png" width=500em />

and whatever else you need to do to debug your code.

Happy debugging!

</exercise>
