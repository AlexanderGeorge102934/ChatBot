# Function: process_transcripts(fname)
# fname: A string indicating a file name
# Returns: Nothing (writes output to file)
#
# This function processes a provided transcript file by creating three versions of it:
# one includes all utterances, with one utterance per line; another includes only the
# chatbot utterances; and the third includes only the user utterances.  None of these
# files should contain the speaker tags ("CHATBOT" or "USER").
def process_transcripts(fname):
    # Open the input file and the three output files
    with open(fname, "r") as f_in, \
            open(f"all_{fname}", "w") as f_out_all, \
            open(f"chatbot_{fname}", "w") as f_out_chatbot, \
            open(f"user_{fname}", "w") as f_out_user:

        current_speaker = None  # Tracks the current speaker

        for line in f_in:
            line = line.strip()  # Remove leading/trailing whitespace

            if not line:
                continue  # Skip empty lines

            # Check if the line indicates a new speaker
            if line.startswith("CHATBOT:"):
                current_speaker = "chatbot"
                # Extract the message after the speaker tag
                message = line[len("CHATBOT:"):].strip()
                if message:
                    f_out_all.write(message + "\n")
                    f_out_chatbot.write(message + "\n")
            elif line.startswith("USER:"):
                current_speaker = "user"
                # Extract the message after the speaker tag
                message = line[len("USER:"):].strip()
                if message:
                    f_out_all.write(message + "\n")
                    f_out_user.write(message + "\n")
            else:
                # Continuation of the previous message
                if current_speaker == "chatbot":
                    f_out_all.write(line + "\n")
                    f_out_chatbot.write(line + "\n")
                elif current_speaker == "user":
                    f_out_all.write(line + "\n")
                    f_out_user.write(line + "\n")
                else:
                    # If no speaker is identified, write to all
                    f_out_all.write(line + "\n")

    print(f"Processed transcripts saved as all_{fname}, chatbot_{fname}, and user_{fname}.")

    return



# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.  If
# you run this code using test.txt as the input file, the contents of the output
# files it produces will be as follows.
#
# all_test.txt:
# Welcome to the CS 421 chatbot!  What is your name?
# Natalie Parde
# Thanks Natalie Parde!  What do you want to talk about today?
# I'm excited that it's a new semester!
# It sounds like you're in a positive mood!
# I'd also like to do a quick stylistic analysis. What's on your mind today?
# I'm currently creating a sample transcript for the CS 421 students.  This will help them ensure that their programs work correctly.  It will also provide an example interaction for them!
# Thanks!  Here's what I discovered about your writing style.
# Type-Token Ratio: 0.8823529411764706
# Average Tokens Per Sentence: 11.333333333333334
# # Nominal Subjects: 5
# # Direct Objects: 2
# # Indirect Objects: 0
# # Nominal Modifiers: 1
# # Adjectival Modifiers: 0
# Custom Feature #1: 0
# Custom Feature #2: 5
# What would you like to do next?  You can quit, redo the sentiment analysis, or redo the stylistic analysis.
# I think I'd like to quit.
#
# chatbot_test.txt:
# Welcome to the CS 421 chatbot!  What is your name?
# Thanks Natalie Parde!  What do you want to talk about today?
# It sounds like you're in a positive mood!
# I'd also like to do a quick stylistic analysis. What's on your mind today?
# Thanks!  Here's what I discovered about your writing style.
# Type-Token Ratio: 0.8823529411764706
# Average Tokens Per Sentence: 11.333333333333334
# # Nominal Subjects: 5
# # Direct Objects: 2
# # Indirect Objects: 0
# # Nominal Modifiers: 1
# # Adjectival Modifiers: 0
# Custom Feature #1: 0
# Custom Feature #2: 5
# What would you like to do next?  You can quit, redo the sentiment analysis, or redo the stylistic analysis.
#
# user_test.txt:
# Natalie Parde
# I'm excited that it's a new semester!
# I'm currently creating a sample transcript for the CS 421 students.  This will help them ensure that their programs work correctly.  It will also provide an example interaction for them!
# I think I'd like to quit.
if __name__ == "__main__":
    fname = "test.txt"
    process_transcripts(fname)