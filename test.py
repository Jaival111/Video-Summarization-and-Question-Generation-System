import re
import json

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda"
MAX_NEW_TOKENS = 350
BATCH_SIZE = 4

t = """### Subjective Questions:
1. What is a notable characteristic of the Flask framework mentioned in the video?
(Answer should be around 1-2 sentences, e.g. "Flask is described as a micro framework that makes it enjoyable to work with back-end web applications.")
### Multiple Choice Questions:
1. At the time of the recording, what version of the Flask framework was released?
   A) 0.9
   B) 1.0
   C) 2.0
   D) 3.0
   Correct Answer: B) 1.0

### Subjective Questions:
1. What is the primary purpose of the application being built in this series of videos?
### Multiple Choice Questions:
1. What is one of the types of posts that users can make in the application?
   A) Only regular blog posts
   B) Only larger, in-depth articles
   C) Regular blog posts, or smaller like Twitter updates
   D) Only videos
   Correct Answer: C) Regular blog posts, or smaller like Twitter updates

### Subjective Questions:
1. Describe the process a user can follow to reset their password if they have forgotten it.
### Multiple Choice Questions:
1. What is one of the new options available to a user after logging in to the application?
   A) Update their email address
   B) Update their password
   C) Update their profile picture
   D) Delete their account
   Correct Answer: C) Update their profile picture

### Subjective Questions:
1. What are some key features of the blog application setup being described, as related to user interaction with posts?
### Multiple Choice Questions:
1. What happens to the picture when a new post is created in the blog application?
   A) It gets deleted.
   B) It gets resized and saved to the server.
   C) It gets updated.
   D) It gets ignored.
   Correct Answer: B) It gets resized and saved to the server.

### Subjective Questions:
1. What are some of the key aspects of building a web application that the instructor mentions as a benefit of building a blog application with Flask?
### Multiple Choice Questions:
1. What can be learned by building a blog application with Flask?
   A) Only how to delete posts
   B) How to deal with databases, user input, file systems, and email sending
   C) How to build a simple chatbot
   D) How to create a game
   Correct Answer: B) How to deal with databases, user input, file systems, and email sending

### Subjective Questions:
1. What is the purpose of the links provided in the description section of the video series?
### Multiple Choice Questions:
1. Where can viewers find the source code of each video in the process?
   A) In the comments section
   B) In the description section below the video
   C) On the website's homepage
   D) On the project's GitHub repository
   Correct Answer: B) In the description section below the video

### Subjective Questions:
1. What is a good practice when installing packages for a new project, as mentioned in the provided text?
(Answer should be around 1-2 paragraphs, demonstrating understanding of the practice of using virtual environments.)
### Multiple Choice Questions:
1. How should you install packages for a new project?
   A) In the default Python environment only.
   B) In a virtual environment, but only for small projects.
   C) In a virtual environment, to keep different projects separate.
   D) It doesn't matter, packages can be installed anywhere.
   Correct Answer: C) In a virtual environment, to keep different projects separate.

### Subjective Questions:
1. What is the approach taken by the educator when starting the series of videos on building a blog application with Flask?
### Multiple Choice Questions:
1. What is the command used to install Flask?
   A) pip install Flask
   B) pip uninstall Flask
   C) pip list Flask
   D) pip update Flask
   Correct Answer: A) pip install Flask


### Subjective Questions:
1. Describe the step you would take to verify that Flask has been installed correctly in the Python interpreter.
### Multiple Choice Questions:
1. Where would you typically create a new project from scratch when following the tutorial on Building a Blog Application with Flask: Features and Setup?      
   A) In the Python interpreter
   B) In a text editor
   C) On the desktop
   D) In a virtual environment
   Correct Answer: C) On the desktop

### Subjective Questions:
1. Describe the command used to create a new directory on a Mac, and explain why it is different from the command used on a Windows machine.
### Multiple Choice Questions:
1. How does the educator create a new project directory in their text editor?
   A) By using the command "mkdir" in the terminal
   B) By navigating to the desktop and creating a new folder
   C) By opening up Sublime Text and selecting "file" > "open"
   D) By using the command "cd" in the terminal
   Correct Answer: C) By opening up Sublime Text and selecting "file" > "open"

### Subjective Questions:
1. What is the name of the new file that the instructor creates within the project directory to start a basic Flask application?
### Multiple Choice Questions:
1. What is the purpose of copying and pasting the code from the Flask website documentation into the newly created file?
   A) To create a new project directory
   B) To start a new Flask application
   C) To view the documentation for the Flask website
   D) To add features to the application
   Correct Answer: B) To start a new Flask application

### Subjective Questions:
1. What is the purpose of the line `from flask import flask` in the code snippet?
### Multiple Choice Questions:
1. What is the reason for passing `__name__` as an argument when creating an instance of the `flask` class?
   A) To specify the name of the application
   B) To import the `flask` class
   C) To pass the name of the module
   D) To create a new instance of the `flask` class
   Correct Answer: C) To pass the name of the module

### Subjective Questions:
1. What is the purpose of using a double underscore name in the context of the script when running it directly with Python?
### Multiple Choice Questions:
1. What is the function of the double underscore name in the script?
   A) To create a new instance of the Flask application
   B) To specify the location of templates and static files
   C) To run the script with Python
   D) To define routes for the application
   Correct Answer: B) To specify the location of templates and static files

### Subjective Questions:
1. What is the recommended approach for learning about decorators in the context of building a Flask application, according to the text?
(Answer should demonstrate understanding of the text's recommendation regarding learning decorators.)
### Multiple Choice Questions:
1. What is the primary purpose of decorators in Flask, as described in the text?
   A) To create new functions from existing ones.
   B) To add additional functionality to existing functions.
   C) To replace existing functions with new ones.
   D) To delete existing functions.
   Correct Answer: B) To add additional functionality to existing functions.

### Subjective Questions:
1. Describe the purpose of the app dot route decorator in the context of building a blog application with Flask.
### Multiple Choice Questions:
1. What is the function of the forward slash in the code snippet provided?
   A) It represents a database connection.
   B) It defines a new route for the website.
   C) It is a placeholder for a variable in the URL.
   D) It is a comment in the code.
   Correct Answer: B) It defines a new route for the website.

### Subjective Questions:
1. What is the initial text that should be displayed when navigating to the homepage of the application?
### Multiple Choice Questions:
1. What command is used to navigate to the project directory in both Windows and Unix-based systems?
   A) CD
   B) MD
   C) LS
   D) CP
   Correct Answer: A) CD

### Subjective Questions:
1. Describe the command used to set an environment variable to the file that is to be the Flask application on a Mac or Linux system.
### Multiple Choice Questions:
1. What is the alternative command used to set an environment variable on a Windows system?
   A) export flask_app = flask_blog.py
   B) set flask_app = flask_blog.py
   C) export flask_app = flask_blog_pi
   D) set flask_app = flask_blog_pi
   Correct Answer: B) set flask_app = flask_blog.py

### Subjective Questions:
1. Describe the IP address that your Flask app is being served on, according to the given text.
### Multiple Choice Questions:
1. What is required to be able to view your Flask site?
   A) A separate server software
   B) A different port number
   C) The web server to be running
   D) A different IP address
   Correct Answer: C) The web server to be running

### Subjective Questions:
1. What is the purpose of using the local host alias instead of the IP address 127.0.0.1 in the context of the given example?
### Multiple Choice Questions:
1. Why is the text "Hello World" displayed in the web browser when the sample application is accessed?
   A) Because it is the default text displayed by the browser.
   B) Because the home route returns this text as a result.
   C) Because the application is not properly configured.
   D) Because the browser is not able to connect to the server.
   Correct Answer: B) Because the home route returns this text as a result.

### Subjective Questions:
1. What is the purpose of wrapping the text in h1 tags in the context of building a blog application with Flask?
### Multiple Choice Questions:
1. What happens when you replace 127 dot zero dot zero dot one with local host in the URL and hit enter?
   A) The route is changed to a different URL
   B) The results of the route are updated to show a different page
   C) The route returns the same results
   D) The application crashes
   Correct Answer: C) The route returns the same results

### Subjective Questions:
1. What is the purpose of wrapping the output in HTML heading one tags in the given code snippet?
### Multiple Choice Questions:
1. Why did the changes made to the code not take effect when the browser was reloaded?
   A) The code was not saved properly.
   B) The web server was not stopped and restarted.
   C) The browser was not updated with the latest version of the code.
   D) The changes were not made in the correct location.
   Correct Answer: B) The web server was not stopped and restarted.

### Subjective Questions:
1. What is a common action you would take when developing a site using the information from the provided text?
### Multiple Choice Questions:
1. What is one way to view the source code of an HTML page in a browser?
   A) By clicking on the "Inspect" button
   B) By right clicking and going to "View Page Source"
   C) By copying and pasting the URL into a code editor
   D) By using a third-party plugin
   Correct Answer: B) By right clicking and going to "View Page Source"

### Subjective Questions:
1. Describe a scenario where having to shut down and restart the web server for every small change in the application would be inconvenient. (Answer should be around 50-75 words)
### Multiple Choice Questions:
1. What is the purpose of setting the environment variable "flask_debug" in the context of running a Flask application?
   A) To enable caching for faster page loads
   B) To set up a database connection
   C) To run the application in debug mode without restarting
   D) To configure email notifications
   Correct Answer: C) To run the application in debug mode without restarting

### Subjective Questions:
1. Describe the command used to set an environment variable on Windows in the context of the provided text.
### Multiple Choice Questions:
1. What is the result of using the "set" command in the context of the provided text?
   A) It sets a new environment variable.
   B) It exports an environment variable.
   C) It runs the Flask application.
   D) It refreshes the browser.
   Correct Answer: A) It sets a new environment variable.

### Subjective Questions:
1. Describe the benefits of running a Flask application in debug mode, as demonstrated in the given text.
### Multiple Choice Questions:
1. What is the purpose of reloading the browser while in debug mode, as shown in the example?
   A) To restart the web server
   B) To update the environment variables
   C) To see the changes made to the application without restarting the web server
   D) To switch to a different programming language
   Correct Answer: C) To see the changes made to the application without restarting the web server

### Subjective Questions:
1. Describe the purpose of the conditional statement that is added to the bottom of the file in the context of building a blog application with Flask.
### Multiple Choice Questions:
1. What is the purpose of the conditional statement that checks if the value of `__name__` is equal to `"__main__"` in the code snippet provided?
   A) To check if the application is running in debug mode
   B) To check if the application is running in production mode
   C) To run the application in debug mode when the value of `__name__` is equal to `"__main__"`
   D) To exit the application when the value of `__name__` is equal to `"__main__"`
   Correct Answer: C) To run the application in debug mode when the value of `__name__` is equal to `"__main__"`

### Subjective Questions:
1. What is the condition that determines whether the name of the double underscore variable is'main' or the name of the module when running a Python script?   
### Multiple Choice Questions:
1. What happens to the name of the double underscore variable when the script is imported into another module?
   A) It remains'main'
   B) It changes to the name of the module
   C) It is undefined
   D) It is set to a random value
   Correct Answer: B) It changes to the name of the module

### Subjective Questions:
1. What are the two ways to run the script in the provided example, and what is the key difference between them?
### Multiple Choice Questions:
1. What happens when the script is run directly with Python by saying "Python and then flask blog dot pi"?
   A) The script is run in a new environment.
   B) The script is run with debug mode enabled.
   C) The script is run with environment variables.
   D) The script is run on a remote server.
   Correct Answer: B) The script is run with debug mode enabled.

### Subjective Questions:
1. What are the two ways to run a Flask application mentioned in the text, and how do they differ in terms of convenience?
### Multiple Choice Questions:
1. Why has the author of the tutorial started using the `flask run` command instead of running the module directly?
   A) Because it is a new feature of Flask that allows for easier debugging.
   B) Because it is more convenient for setting environment variables.
   C) Because the documentation now recommends it.
   D) Because it is a more secure way to run the application.
   Correct Answer: C) Because the documentation now recommends it.

### Subjective Questions:
1. Describe the purpose of adding an about route to the existing Flask application.
### Multiple Choice Questions:
1. What was the response returned by the server when trying to navigate to the about page before adding the about route?
   A) 200 OK
   B) 404 Not Found
   C) 500 Internal Server Error
   D) 301 Moved Permanently
   Correct Answer: B) 404 Not Found

### Subjective Questions:
1. What is the purpose of creating a new route in the context of building a blog application with Flask?
### Multiple Choice Questions:
1. What change is necessary when creating a new route in the Flask application?
   A) Change the route path but keep the function name the same.
   B) Change the function name but keep the route path the same.
   C) Change both the route path and function name.
   D) No change is necessary.
   Correct Answer: B) Change the function name but keep the route path the same.

### Subjective Questions:
1. Describe the change made to the route in the code to resolve the 404 not found error.
### Multiple Choice Questions:
1. What was the purpose of the new route created in the code?
   A) To handle user authentication
   B) To serve static files
   C) To return the information for the about page
   D) To handle form submissions
   Correct Answer: C) To return the information for the about page

### Subjective Questions:
1. Describe the purpose of using multiple decorators in Flask to handle multiple routes with the same function.
### Multiple Choice Questions:
1. What is the purpose of using multiple decorators in the provided example?
   A) To handle multiple routes with different functions.
   B) To handle multiple routes with the same function.
   C) To handle multiple functions with the same route.
   D) To handle multiple routes with different HTTP methods.
   Correct Answer: B) To handle multiple routes with the same function.

### Subjective Questions:
1. What is the effect of changing the function name from "hello" to "home" in the code snippet described in the text?
### Multiple Choice Questions:
1. What is the outcome of adding a new route "/home" to the Flask application described in the text?
   A) The "/home" route will override the existing "/home" route.
   B) The "/home" route will be handled by a new function.
   C) The "/home" route will be handled by the same function as the existing route.
   D) The "/home" route will not be recognized by the Flask application.
   Correct Answer: C) The "/home" route will be handled by the same function as the existing route.

### Subjective Questions:
1. What is the next step in the tutorial series after learning how to create basic routes in a Flask application?
### Multiple Choice Questions:
1. What will be covered in the next video of the tutorial series?
   A) How to handle user authentication in Flask
   B) How to return some more complicated HTML code using templates
   C) How to integrate a database with Flask
   D) How to deploy a Flask application to a production environment
   Correct Answer: B) How to return some more complicated HTML code using templates

### Subjective Questions:
1. What is one way viewers can contribute to the content creator's work if they have the means?
### Multiple Choice Questions:
1. How can viewers help share the videos with others?
   A) By commenting on the videos
   B) By sharing the link on social media
   C) By subscribing to the channel
   D) By contributing through Patreon
   Correct Answer: B) By sharing the link on social media"""

def parse_questions_from_output(output_text: str, chunk_index: int) -> dict:
    """Parse questions from a single output text."""
    questions = {
        "chunk_index": chunk_index,
        "subjective_question": "",
        "multiple_choice_question": "",
        "mcq_options": [],
        "correct_answer_index": -1
    }
    
    # Split by sections
    sections = output_text.split("###")
    
    for section in sections:
        section = section.strip()
        if "Subjective Questions:" in section:
            # Extract subjective question
            lines = section.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Subjective Questions:"):
                    # Remove numbering and reference
                    question = re.sub(r'^\d+\.\s*', '', line)
                    question = re.sub(r'\s*\(Reference: Chunk \d+\)', '', question)
                    if question:
                        questions["subjective_question"] = question
                        break
        
        if "Multiple Choice Questions:" in section:
            # Extract multiple choice question
            lines = section.split("\n")
            current_question = ""
            options = []
            correct_answer = ""
            
            for line in lines:
                line = line.strip()
                if line and not line.startswith("Multiple Choice Questions:"):
                    if re.match(r'^\d+\.', line):
                        # Question
                        current_question = re.sub(r'^\d+\.\s*', '', line)
                        current_question = re.sub(r'\s*\(Reference: Chunk \d+\)', '', current_question)
                    elif re.match(r'^[A-D]\)', line):
                        # Option
                        options.append(line)
                    elif line.startswith("Correct Answer:"):
                        # Correct answer
                        correct_answer = line.replace("Correct Answer:", "").strip()
                        correct_answer = re.sub(r'\s*\(Reference: Chunk \d+\)', '', correct_answer)
            
            questions["multiple_choice_question"] = current_question
            questions["mcq_options"] = options
            
            # Find correct answer index
            if correct_answer:
                for i, option in enumerate(options):
                    if option.strip() == correct_answer.strip():
                        questions["correct_answer_index"] = i
                        break
    
    return questions

def save_questions_to_json(questions_list: list, output_file: str = "generated_questions.json"):
    """Save questions to JSON file."""
    import datetime
    
    # Create the final structure
    output_data = {
        "metadata": {
            "generated_at": datetime.datetime.now().isoformat(),
            "model_used": MODEL_NAME,
            "device": DEVICE,
            "total_chunks": len(questions_list),
            "total_questions": len(questions_list) * 2  # 1 subjective + 1 MCQ per chunk
        },
        "questions": questions_list
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"Questions saved to: {output_file}")

lis = t.split("\n\n")
# print(lis)
questions_list = []
for idx, output in enumerate(lis):
    parsed_questions = parse_questions_from_output(output, idx + 1)  # chunk_index starts from 1
    questions_list.append(parsed_questions)

save_questions_to_json(questions_list)