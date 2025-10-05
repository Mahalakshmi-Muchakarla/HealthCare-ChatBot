from flask import Flask, render_template, request, jsonify, session
import sys
import os
import uuid  
from typing import List
import numpy as np 
import pandas as pd # Ensure pandas is imported if chat_bot is using it (which it is)

# Add parent directory to path so we can import chat_bot.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import chat_bot 

# --- FLASK SETUP ---
app = Flask(
    __name__,
    template_folder='../frontend',
    static_folder='../frontend'     
)
app.secret_key = str(uuid.uuid4())

# --- CHATBOT STATE MANAGEMENT ---
STATE_GET_NAME = "get_name"       
STATE_GET_AGE = "get_age"         
STATE_GET_SYMPTOM = "get_symptom"
STATE_GET_SYMPTOM_SELECTION = "get_symptom_selection" 
STATE_GET_DAYS = "get_days"
STATE_GET_OTHER_SYMPTOMS = "get_other_symptoms" # The iterative loop state
STATE_CLOSED = "session_closed"   

# --- FLASK ROUTES ---

@app.route('/')
def index():
    # Clear and initialize session state for a fresh start
    session.clear() 
    session['chat_state'] = STATE_GET_NAME
    session['symptoms_exp'] = []
    session['initial_prediction'] = None 
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_message = request.json.get("message").strip()
    current_state = session.get('chat_state', STATE_GET_NAME)
    bot_reply = ""
    # print(f"DEBUG: Current State: {current_state}, User Message: {user_message}") # You can use this for debugging

    # --- STATE 1: GET NAME ---
    if current_state == STATE_GET_NAME:
        if not user_message or len(user_message.strip()) < 2:
            bot_reply = "Please enter your name to start."
            return jsonify({"response": bot_reply})
            
        session['user_name'] = user_message.strip()
        session['chat_state'] = STATE_GET_AGE
        bot_reply = f"Thank you, {session['user_name']}. **What is your age**? (Please enter a number)"
        
    # --- STATE 2: GET AGE ---
    elif current_state == STATE_GET_AGE:
        try:
            user_age = int(user_message)
            session['user_age'] = user_age
            session['chat_state'] = STATE_GET_SYMPTOM
            bot_reply = f"Got it, {user_age} years old. Please **enter the main symptom** you are experiencing."
        except ValueError:
            bot_reply = "Invalid input. Please enter your age as **a valid number**."

            
    # --- STATE 3: GET MAIN SYMPTOM & LOOK UP OPTIONS ---
    elif current_state == STATE_GET_SYMPTOM:
        
        chk_dis: List[str] = ",".join(chat_bot.cols).split(",")
        conf, cnf_dis = chat_bot.check_pattern(chk_dis, user_message)

        if conf == 1:
            if len(cnf_dis) > 1:
                session['symptom_options'] = cnf_dis
                options_str = "<br>".join([f"{i}) {it.replace('_', ' ')}" for i, it in enumerate(cnf_dis)])
                session['chat_state'] = STATE_GET_SYMPTOM_SELECTION
                bot_reply = f"I found multiple matches. Please **select the one you meant (0 - {len(cnf_dis) - 1})**:<br>{options_str}"
            else:
                session['symptoms_exp'] = [cnf_dis[0]]
                session['chat_state'] = STATE_GET_DAYS
                bot_reply = f"Understood, **{cnf_dis[0].replace('_', ' ')}**. From how many **days** have you been experiencing this? (Please enter **just the number**, e.g., 5)"
        else:
            bot_reply = "I couldn't find that symptom. Please enter a valid symptom name."

    # --- STATE 3A: HANDLE SYMPTOM SELECTION ---
    elif current_state == STATE_GET_SYMPTOM_SELECTION:
        try:
            selection_index = int(user_message)
            cnf_dis = session.get('symptom_options')
            
            if cnf_dis and 0 <= selection_index < len(cnf_dis):
                selected_symptom = cnf_dis[selection_index]
                session['symptoms_exp'] = [selected_symptom]
                session['chat_state'] = STATE_GET_DAYS
                bot_reply = f"Got it: **{selected_symptom.replace('_', ' ')}**. From how many **days** have you been experiencing this? (Please enter **just the number**, e.g., 5)"
            else:
                bot_reply = "Invalid selection. Please enter the number corresponding to the symptom."
        except ValueError:
            bot_reply = "Please enter a valid number for your selection."

    # --- STATE 4: GET DAYS & INITIATE ITERATIVE QUESTIONING ---
    elif current_state == STATE_GET_DAYS:
        try:
            num_days = int(user_message)
            session['num_days'] = num_days
            session['chat_state'] = STATE_GET_OTHER_SYMPTOMS 
            
            # --- TERMINAL-STYLE LOGIC FOR QUESTION EXTRACTION ---
            primary_symptom = session['symptoms_exp'][0]
            
            # 1. Get the initial prediction (equivalent to 'present_disease')
            predicted_disease_arr = chat_bot.sec_predict([primary_symptom])
            initial_prediction = predicted_disease_arr[0]
            session['initial_prediction'] = initial_prediction 
            
            # 2. Extract ALL associated symptoms from the training data (reduced_data)
            symptoms_to_ask_list = []
            
            try:
                reduced_data = chat_bot.reduced_data
                if initial_prediction in reduced_data.index:
                    disease_data_row = reduced_data.loc[initial_prediction].values[0]
                    # Check if the data is a DataFrame or Series that supports .nonzero()
                    if isinstance(disease_data_row, np.ndarray) and disease_data_row.ndim == 2:
                        # Assuming it's a 2D array if .values was used on a Series/DataFrame row
                        disease_data_row = disease_data_row[0]

                    nonzero_indices = np.where(disease_data_row == 1)[0]
                    symptoms_given = reduced_data.columns[nonzero_indices]
                    
                    # Filter out the primary symptom
                    symptoms_to_ask_list = [
                        sym for sym in list(symptoms_given) 
                        if sym != primary_symptom
                    ]
                
            except Exception as e:
                # print(f"DEBUG: Error during symptom extraction: {e}")
                # FALLBACK: If lookup fails, try to use ALL symptoms as a fallback for thoroughness
                # This is a very rough guess but ensures we ask multiple questions
                all_symptoms = [c for c in chat_bot.cols if c not in session['symptoms_exp']]
                symptoms_to_ask_list = all_symptoms[:10] 
            
            session['symptoms_to_ask'] = symptoms_to_ask_list
            session['current_symptom_index'] = 0
            
            # 3. Start the iterative questioning or conclude if no other symptoms exist
            if session['symptoms_to_ask']:
                first_secondary_symptom = session['symptoms_to_ask'][0].replace('_', ' ')
                bot_reply = f"Thank you. Now, are you experiencing any related symptoms? For example, are you experiencing **{first_secondary_symptom}**? (Yes/No)"
            else:
                # Fallback: If no secondary questions are found, give the conclusion immediately.
                session['chat_state'] = STATE_CLOSED 
                
                description = chat_bot.description_list.get(initial_prediction, "No description available.")
                precautions = chat_bot.precautionDictionary.get(initial_prediction, [])
                prec_list_html = "".join([f"<li>{p}</li>" for p in precautions])
                
                bot_reply = f"Based on the symptom you gave, I predict **{initial_prediction}**.<br><br>Description: {description}<br><br>Take the following measures:<ol>{prec_list_html}</ol>."
                bot_reply += "<br><br>**Consult a nearby doctor if pain persists or symptoms worsen.**" 
                bot_reply += "<br><br>Type **start** to begin a new session." 
            
        except ValueError:
            session['chat_state'] = STATE_GET_DAYS 
            bot_reply = "Invalid input. Please enter **only a number** for the days (e.g., 5)."

    # --- STATE 5: GET OTHER SYMPTOMS (The Iterative Loop & Conclusion) ---
    elif current_state == STATE_GET_OTHER_SYMPTOMS:
        user_input = user_message.lower().strip()
        symptoms_to_ask = session.get('symptoms_to_ask', [])
        current_index = session.get('current_symptom_index', 0)
        
        # Check if the list of questions is valid and we haven't answered all of them yet
        if symptoms_to_ask and current_index < len(symptoms_to_ask):
            
            current_symptom = symptoms_to_ask[current_index]

            # 1. VALIDATION: Check for invalid input (re-ask the SAME question)
            if user_input not in ["yes", "no"]:
                current_symptom_display = current_symptom.replace('_', ' ')
                bot_reply = f"Please answer with **Yes** or **No**. Are you experiencing **{current_symptom_display}**?"
                return jsonify({"response": bot_reply})

            # 2. PROCESSING: Process the previous question's answer
            if user_input == "yes":
                session['symptoms_exp'].append(current_symptom)

            # 3. TRANSITION: Move to the next question index
            next_index = current_index + 1
            session['current_symptom_index'] = next_index
            
            if next_index < len(symptoms_to_ask):
                # CONTINUE QUESTIONING: Ask the next question
                next_symptom_display = symptoms_to_ask[next_index].replace('_', ' ')
                bot_reply = f"Next: Are you also experiencing **{next_symptom_display}**? (Yes/No)"
                return jsonify({"response": bot_reply}) # EXIT HERE TO ASK NEXT QUESTION

        # --- FINAL CONCLUSION LOGIC (Runs when the last question's answer is processed or list was empty) ---
        
        # 4. FINAL PREDICTION: All questions asked, generate the dual result
        final_symptoms = session['symptoms_exp']
        initial_prediction = session.get('initial_prediction') # The 'present_disease'
        
        # Final prediction based on ALL collected symptoms (the 'second_prediction')
        final_prediction_arr = chat_bot.sec_predict(final_symptoms)
        final_disease = final_prediction_arr[0]
        
        # 1. Determine the output diseases (Dual Prediction Logic)
        disease_list = [initial_prediction]
        if initial_prediction != final_disease:
            disease_list.append(final_disease)
        
        # Format the disease names for the output
        disease_names_html = " or ".join([f"**{d}**" for d in disease_list])
        disease_output = f"It might not be that bad but you should take precautions.<br>You may have {disease_names_html}."
        
        # 2. Build the description output (show description for all predicted diseases)
        description_output = "Description:<br>"
        for disease in disease_list:
            desc = chat_bot.description_list.get(disease, f"No description available for **{disease}**.")
            description_output += f"**{disease}**: {desc.replace('_', ' ')}<br><br>"
        
        # 3. Get precautions (based on the initial prediction, matching terminal bot)
        precautions = chat_bot.precautionDictionary.get(initial_prediction, [])
        prec_list_html = "".join([f"<li>{p.replace('_', ' ')}</li>" for p in precautions])

        
        bot_reply = f"{disease_output}<br><br>{description_output}Take the following measures:<ol>{prec_list_html}</ol>."
        
        # Add the final safety disclaimer
        bot_reply += "<br><br>**Consult a nearby doctor if pain persists or symptoms worsen.**" 
        
        session['chat_state'] = STATE_CLOSED 
        bot_reply += "<br><br>Type **start** to begin a new session."


    # --- STATE 6: CLOSED ---
    elif current_state == STATE_CLOSED: 
        if user_message.lower() == "start":
            # Reset the session to begin again
            session.clear() 
            session['chat_state'] = STATE_GET_NAME 
            bot_reply = "Starting a new session. Hello! I am the Healthcare Bot 🤖. Before we begin, **What is your name?**"
        else:
            bot_reply = "Session complete. Please type **start** to begin a new session."
            
    return jsonify({"response": bot_reply})

# Run the app
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)