# WhatsApp-Analyser
This is a full fledged whatsapp analyser intended to work on any conversation.

My aim is to build an app which can be used by anyone in any country typing in any language. This is just the initial release wherein I have built the following features:
  * Most used words in a conversation
  * Peak detection on the word-frequency graph (just for the kicks. Doesn't accomplish anything more than the above feature.)
  * The most and least talkative people in a group. Create a pie chart of individual % contribution to messages.
  * An activity map to analyse during what time of the day is the group most active.
  * Most frequently used words by individual participants.
  * Conversation trends over full timeline.

Note that this project is far from complete. I would love to receive new ideas and improvements or bug fixes/reports. Please report any problem you encounter while using this script.

======================

### To use this script:
 1. Email the conversation to yourself (options->more->email chat or simply long press the desired group and then email chat) without media.
 2. Save this textfile in the same folder as this scipt (or enter full file path in the next step.)
 3. You will be prompted to enter the name of the conversation with extension. Enter the name and see the results.
 4. The results are best viewed in an ipython notebook. I used [Jupyter IPython Notebook](http://jupyter.readthedocs.org/en/latest/install.html#how-to-install-jupyter-notebook).

======================

### Following are the module dependencies:
 * [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
 * [matplotlib](http://matplotlib.org/users/installing.html)
