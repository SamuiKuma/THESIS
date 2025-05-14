import sys
import os
import time
import re
import flet as ft
import matplotlib.pyplot as plt  # Import for visualization
import base64  # Import for encoding visualization images
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from audio_processing import is_valid_audio, extract_features
from lessonScore import lesson_score
# Replace or comment out these imports
# from speech_processor import recognize_from_microphone, enhance_audio_for_waray
# Add the new import
from speech_recognition_utils import SpeechProcessor, capture_audio  # Import the more complete module

correct_answers = {}
incorrect_answers = {}
grade_percentage = 0.0
total_response_time = 0.0
formatted_time = ""

def get_questions(page):
    """Retrieves questions for the current lesson."""
    level_data = page.session.get("level_data")
    if not level_data:
        print("Level data not found in session.")
        return None

    questions = level_data.questions_answers
    
    # TESTING MODE: Filter only questions with ID attribute value 5
    filtered_questions = []
    for q in questions:
        # Print the object to debug
        print(f"Question object: {q.__dict__}")
        # Try to access the potential ID attributes
        question_id = getattr(q, "id", None)
        if question_id == 5:
            filtered_questions.append(q)
            print(f"Found question with ID 5: {q.__dict__}")
    
    # If we found ID 5 questions, use them, otherwise use all questions
    if filtered_questions:
        questions = filtered_questions
        print(f"Filtered to {len(questions)} questions with ID 5")
    
    if not questions:
        print("No questions found.")
        return None

    for q in questions:
        q.lesson_id = level_data.lesson_id
        q.module_name = level_data.module_name

    return questions

def build_lesson_question(question_data, progress_value, on_next, on_back):
    """Builds the layout for a 'Lesson' type question."""
    start_time = time.time()
    header = "Lesson"
    waray_phrase = None
    english_translation = None
    full_definition = question_data.question
    question = full_definition

    def add_time (e):
        global total_response_time
        response_time = time.time() - start_time
        question_data.response_time = response_time
        total_response_time += response_time

        if on_next:
            on_next(e)

    # Find all substrings in single quotes
    matches = re.findall(r"'(.*?)'", question)

    if len(matches) >= 2:
        waray_phrase = matches[0]
        english_translation = matches[1]
    else:
        print("Not enough matches found in the question string.")

    card_content = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(ft.Divider(color="grey", thickness=1), width=60),
                        ft.Container(
                            ft.Text(header, color="grey", size=14, weight=ft.FontWeight.W_500),
                            padding=ft.padding.symmetric(horizontal=10)
                        ),
                        ft.Container(ft.Divider(color="grey", thickness=1), width=60),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                ft.Row(
                    [
                        ft.IconButton(
                            icon=ft.icons.VOLUME_UP,
                            icon_color="#0078D7",
                            icon_size=24,
                            # Optionally play audio here
                        ),
                        ft.Text(
                            waray_phrase,
                            color="#0078D7",
                            size=24,
                            weight=ft.FontWeight.BOLD
                        )
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=5
                ),
                ft.Container(
                    ft.Text(english_translation, color="grey", size=16),
                    margin=ft.margin.only(bottom=20)
                ),
                ft.Container(
                    ft.Image(
                        src="assets/L1.png",
                        width=250,
                        height=150,
                        fit=ft.ImageFit.CONTAIN
                    ),
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=20)
                ),
                ft.Container(
                    ft.Text(
                        full_definition,
                        text_align=ft.TextAlign.CENTER,
                        size=16,
                        weight=ft.FontWeight.W_500,
                        color="black"
                    ),
                    margin=ft.margin.only(bottom=20)
                )
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10
        ),
        width=312,
        bgcolor="white",
        border_radius=10,
        padding=20,
        margin=ft.margin.only(top=20, bottom=20)
    )

    progress = ft.Container(
        ft.ProgressBar(value=progress_value, bgcolor="#e0e0e0", color="#0078D7", width=300),
        margin=ft.margin.only(bottom=20)
    )

    bottom_nav = ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        icon_color="grey",
                        on_click=on_back
                    ),
                    width=100,
                    bgcolor="white",
                    border_radius=ft.border_radius.all(30),
                    padding=5
                ),
                ft.Container(width=10),
                ft.Container(
                    content=ft.ElevatedButton(
                        content=ft.Text("NEXT", color="white", weight=ft.FontWeight.BOLD, size=16),
                        style=ft.ButtonStyle(
                            bgcolor={"": "#0078D7"},
                            shape=ft.RoundedRectangleBorder(radius=30),
                        ),
                        width=200,
                        height=50,
                        on_click=add_time
                    )
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER
        ),
        padding=ft.padding.only(bottom=20)
    )

    return ft.Column(
        [
            card_content,
            progress,
            bottom_nav
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )

def build_pronounce_question(question_data, progress_value, on_next, on_back):
    start_time = time.time()
    question_text = question_data.question
    vocabulary = question_data.vocabulary
    accuracy_threshold = getattr(question_data, 'accuracy_threshold', 0.75)
    
    # Remove the incorrect Page._current reference
    # Instead, we'll use the page reference from the update function context
    
    # Create speech processor with error handling
    try:
        # Use explicit paths to ensure files are found
        script_dir = os.path.dirname(os.path.abspath(__file__))
        proj_dir = os.path.abspath(os.path.join(script_dir, '../../'))
        model_path = os.path.join(proj_dir, 'waray_speech_model.keras')
        encoder_path = os.path.join(proj_dir, 'encoder_classes.npy')
        
        speech_processor = SpeechProcessor(model_path=model_path, encoder_path=encoder_path)
        model_available = speech_processor.model is not None
    except Exception as e:
        print(f"Error loading speech processor: {str(e)}")
        model_available = False
        speech_processor = None

    recording = {"is_recording": False, "audio_data": None, "file_path": None}
    transcription = {"text": "", "accuracy": 0.0}

    # Create UI components
    txt_transcription = ft.Text("Tap the microphone to start recording", color="grey", size=16)
    txt_accuracy = ft.Text("", size=16)
    pronunciation_tips = ft.Text("", size=14, color="orange", visible=False)
    pronunciation_chart = ft.Image(visible=False)
    button_mic = ft.IconButton(
        icon=ft.icons.MIC,
        icon_color="white",
        bgcolor="#0078D7",
        icon_size=36,
        on_click=lambda e: start_recording(e)
    )
    
    # Import threading here to avoid issues
    import threading

    def start_recording(e):
        button_mic.disabled = True
        txt_transcription.value = "Listening..."
        txt_accuracy.value = ""
        pronunciation_tips.visible = False
        pronunciation_chart.visible = False
        e.page.update()
        
        recording["is_recording"] = True
        threading.Thread(target=lambda: record_audio(e.page)).start()

    def record_audio(page):
        if not model_available:
            # Simulate audio processing when model isn't available
            time.sleep(2)
            txt_transcription.value = vocabulary  # Assume correct for demo
            txt_accuracy.value = "Model not available - simulating correct pronunciation"
            txt_accuracy.color = "orange"
            button_mic.disabled = False
            button_mic.bgcolor = "#0078D7"
            button_mic.icon_color = "white"
            page.update()
            return
            
        try:
            recording["file_path"] = capture_audio(duration=3)
            
            if recording["file_path"] and os.path.exists(recording["file_path"]):
                process_recording(page)
            else:
                txt_transcription.value = "No audio detected. Please try again."
                txt_accuracy.value = ""
                button_mic.disabled = False
                page.update()
        except Exception as e:
            txt_transcription.value = f"Error recording audio: {str(e)}"
            button_mic.disabled = False
            page.update()
        finally:
            recording["is_recording"] = False
            
    def process_recording(page):
        if not model_available:
            return
            
        try:
            predicted_word, confidence, phoneme_confidence = speech_processor.predict_speech(
                recording["file_path"], vocabulary
            )
            
            if predicted_word:
                txt_transcription.value = f"You said: {predicted_word}"
                
                # Get any pronunciation errors from the NLTK analysis that was performed
                nltk_errors = getattr(speech_processor, 'pronunciation_errors', [])
                
                if predicted_word.lower() == vocabulary.lower():
                    accuracy = confidence if confidence else 0.75
                    txt_accuracy.value = f"Accuracy: {accuracy:.0%}"
                    
                    if accuracy >= accuracy_threshold:
                        txt_accuracy.color = "green"
                        question_data.accuracy = accuracy
                        
                        # Show detailed phoneme feedback
                        if phoneme_confidence:
                            # Identify problematic phonemes
                            problem_phonemes = [(p, s) for p, s in phoneme_confidence.items() if s < 0.7]
                            if problem_phonemes:
                                feedback_text = "Work on: "
                                feedback_text += ", ".join([f"{p} ({s:.0%})" for p, s in problem_phonemes])
                                
                                # Add NLTK analysis if available
                                if nltk_errors:
                                    feedback_text += "\n\nGoogle analysis: " + "\n• ".join([""] + nltk_errors)
                                    
                                pronunciation_tips.value = feedback_text
                                pronunciation_tips.visible = True
                            else:
                                pronunciation_tips.visible = False
                                
                            # Generate and display visualization
                            viz_buffer = visualize_pronunciation_feedback(vocabulary, phoneme_confidence)
                            if viz_buffer:
                                pronunciation_chart.src_base64 = base64.b64encode(viz_buffer.read()).decode('utf-8')
                                pronunciation_chart.visible = True
                    else:
                        txt_accuracy.color = "orange"
                        question_data.accuracy = accuracy
                        
                        # Show pronunciation tips for specific syllables
                        if phoneme_confidence:
                            problem_syllables = speech_processor._identify_problem_syllables(
                                [(p, s) for p, s in phoneme_confidence.items() if s < 0.7],
                                speech_processor._map_phonemes_to_syllables(vocabulary.lower())
                            )
                            
                            feedback_text = ""
                            if problem_syllables:
                                feedback_text = f"Focus on syllables: {', '.join(problem_syllables)}"
                            
                            # Add NLTK analysis if available
                            if nltk_errors:
                                if feedback_text:
                                    feedback_text += "\n\nGoogle analysis: " + "\n• ".join([""] + nltk_errors)
                                else:
                                    feedback_text = "Google analysis: " + "\n• ".join([""] + nltk_errors)
                            
                            pronunciation_tips.value = feedback_text
                            pronunciation_tips.visible = bool(feedback_text)
                else:
                    txt_transcription.value = f"You said: {predicted_word}. Try saying '{vocabulary}'"
                    txt_accuracy.value = f"Incorrect word detected"
                    txt_accuracy.color = "red"
                    question_data.accuracy = 0.0
                    
                    # Show general pronunciation tips with NLTK analysis
                    feedback_text = "Try again, focusing on clear pronunciation"
                    
                    if nltk_errors:
                        feedback_text += "\n\nPronunciation analysis: " + "\n• ".join([""] + nltk_errors)
                    
                    pronunciation_tips.value = feedback_text
                    pronunciation_tips.visible = True
                    pronunciation_chart.visible = False
            else:
                txt_transcription.value = "Speech not recognized clearly. Please try again."
                txt_accuracy.value = ""
                question_data.accuracy = 0.0
                pronunciation_tips.visible = False
                pronunciation_chart.visible = False
                
        except Exception as e:
            txt_transcription.value = f"Error processing speech: {str(e)}"
            txt_accuracy.value = ""
            pronunciation_tips.visible = False
            pronunciation_chart.visible = False
            
        finally:
            button_mic.disabled = False
            page.update()
            
            # Clean up temp file
            try:
                if recording["file_path"] and os.path.exists(recording["file_path"]):
                    os.remove(recording["file_path"])
            except Exception:
                pass

    def handle_next(e):
        response_time = time.time() - start_time
        question_data.response_time = response_time
        global total_response_time
        total_response_time += response_time
        
        if not question_data.accuracy:
            question_data.accuracy = 0.0
            
        if question_data.accuracy >= accuracy_threshold:
            print(f"Pronunciation accepted with accuracy: {question_data.accuracy:.2f}")
            correct_answers[question_data.question] = question_data
        else:
            print(f"Pronunciation below threshold: {question_data.accuracy:.2f}")
            incorrect_answers[question_data.question] = question_data
            
        if on_next:
            on_next(e)
    
    # Create the main UI layout
    card_content = ft.Container(
        content=ft.Column(
            [
                ft.Row(
                    [
                        ft.Container(ft.Divider(color="grey", thickness=1), width=60),
                        ft.Container(
                            ft.Text("Pronounce", color="grey", size=14, weight=ft.FontWeight.W_500),
                            padding=ft.padding.symmetric(horizontal=10)
                        ),
                        ft.Container(ft.Divider(color="grey", thickness=1), width=60),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER
                ),
                ft.Container(
                    ft.Text(
                        question_text,
                        text_align=ft.TextAlign.CENTER,
                        size=16,
                        weight=ft.FontWeight.W_500
                    ),
                    margin=ft.margin.only(bottom=10, top=10)
                ),
                ft.Container(
                    ft.Text(
                        vocabulary,
                        color="#0078D7",
                        size=28,
                        weight=ft.FontWeight.BOLD,
                        text_align=ft.TextAlign.CENTER
                    ),
                    margin=ft.margin.only(bottom=20)
                ),
                ft.Container(
                    button_mic,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=20, top=10)
                ),
                ft.Container(
                    txt_transcription,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    txt_accuracy,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    pronunciation_tips,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=10)
                ),
                ft.Container(
                    pronunciation_chart,
                    alignment=ft.alignment.center,
                    margin=ft.margin.only(bottom=20)
                ),
            ],
            alignment=ft.MainAxisAlignment.START,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5
        ),
        width=312,
        bgcolor="white",
        border_radius=10,
        padding=20,
        margin=ft.margin.only(top=20, bottom=20)
    )

    progress = ft.Container(
        ft.ProgressBar(value=progress_value, bgcolor="#e0e0e0", color="#0078D7", width=300),
        margin=ft.margin.only(bottom=20)
    )

    bottom_nav = ft.Container(
        content=ft.Row(
            [
                ft.Container(
                    content=ft.IconButton(
                        icon=ft.icons.ARROW_BACK,
                        icon_color="grey",
                        on_click=on_back
                    ),
                    width=100,
                    bgcolor="white",
                    border_radius=ft.border_radius.all(30),
                    padding=5
                ),
                ft.Container(width=10),
                ft.Container(
                    content=ft.ElevatedButton(
                        content=ft.Text("NEXT", color="white", weight=ft.FontWeight.BOLD, size=16),
                        style=ft.ButtonStyle(
                            bgcolor={"": "#0078D7"},
                            shape=ft.RoundedRectangleBorder(radius=30),
                        ),
                        width=200,
                        height=50,
                        on_click=handle_next
                    )
                )
            ],
            alignment=ft.MainAxisAlignment.CENTER
        ),
        padding=ft.padding.only(bottom=20)
    )

    return ft.Column(
        [
            card_content,
            progress,
            bottom_nav
        ],
        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        expand=True
    )

def visualize_pronunciation_feedback(word, phoneme_confidence):
    """Generate a visual representation of pronunciation accuracy for each phoneme."""
    if not phoneme_confidence:
        return None
        
    # Create figure
    fig, ax = plt.figure(figsize=(10, 3)), plt.gca()
    
    # Colors for different confidence levels
    colors = ['#ff6b6b', '#ffa06b', '#ffd46b', '#d4ff6b', '#6bff6b']
    
    # Create bars for each phoneme
    phonemes = list(phoneme_confidence.keys())
    scores = list(phoneme_confidence.values())
    
    # Create bars with color gradients based on score
    bars = ax.bar(phonemes, scores, color=[colors[min(int(s*5), 4)] for s in scores])
    
    # Add labels
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Pronunciation Analysis for '{word}'")
    ax.set_ylabel("Confidence Score")
    ax.set_xlabel("Phonemes")
    
    # Add threshold line
    ax.axhline(y=0.7, linestyle='--', color='gray', alpha=0.7)
    ax.text(len(phonemes)/2, 0.72, "Acceptable Threshold", ha='center', va='bottom', color='gray')
    
    # Add problem indicators
    for i, score in enumerate(scores):
        if score < 0.7:
            ax.text(i, score + 0.05, "!", ha='center', va='bottom', color='red', fontweight='bold')
    
    # Save to buffer
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return buf  # Return buffer for display in GUI

def render_question_layout(question_data, progress_value, on_next, on_back):
    """Renders the appropriate question layout based on question type."""
    question_type = question_data.type
    print(f"Rendering question type: {question_type}")

    if question_type == "Lesson":
        return build_lesson_question(question_data, progress_value, on_next, on_back)
    elif question_type == "Image Picker":
        return build_imgpicker_question(question_data, progress_value, on_next, on_back)
    elif question_type == "Word Select / Translate":
        return build_wordselect_question(question_data, progress_value, on_next, on_back)
    elif question_type == "True or False":
        return build_tf_question(question_data, progress_value, on_next, on_back)
    elif question_type == "Cultural Trivia":
        return build_trivia_question(question_data, progress_value, on_next, on_back)
    elif question_type == "Pronounce":
        print("Using Pronounce question layout")
        return build_pronounce_question(question_data, progress_value, on_next, on_back)
    else:
        # Default to word select if type is unknown
        print(f"Unknown question type: {question_type}. Using Word Select Layout")
        return build_wordselect_question(question_data, progress_value, on_next, on_back)

def lesson_page(page: ft.Page):
    page.title = "Arami - Lesson"
    page.padding = 0

    def go_back(e):
        page.go("/levels")

    # Background with landscape image
    background = ft.Container(
        content=ft.Image(
            src="assets/landscape_background.png",
            width=page.width,
            height=page.height,
            fit=ft.ImageFit.COVER
        ),
        expand=True
    )
    
    # Load questions
    questions = get_questions(page)
    total_questions = len(questions)
    weighted_questions = [q for q in questions if getattr(q, 'correct_answer', None) is not None]
    current_question_index = {"value": 0}
    progress_value = (current_question_index["value"] + 1) / total_questions
    if not questions:
        page.views.append(ft.View("/lesson", [ft.Text("No questions available.")]))
        return

    def render_current_question(progress_value):
        page.views.clear()  # Optional: clear previous view
        question = questions[current_question_index["value"]]
        content = render_question_layout(
            question_data=question,
            progress_value=progress_value,
            on_next=next_question,
            on_back=go_back
        )

        page.views.append(
            ft.View(
                "/lesson",
                [ft.Stack([background, content], expand=True)],
                padding=0
            )
        )
        page.update()
    
    def next_question(e=None):
        current_question_index["value"] += 1
        progress_value = (current_question_index["value"] + 1) / total_questions
        if current_question_index["value"] < len(questions):
            render_current_question(progress_value)
        else:
            print(len(weighted_questions))
            print(len(correct_answers))
            grade_percentage = (len(correct_answers) / len(weighted_questions)) * 100
            formatted_time = f"{int(total_response_time // 60)}:{int(total_response_time % 60):02d}"
            page.session.set("updated_data", [grade_percentage, formatted_time, correct_answers, incorrect_answers, questions])
            lesson_score(page, grade_percentage, correct_answers, incorrect_answers, formatted_time)
            reset_var()

    def reset_var():
        current_question_index["value"] = 0
        global correct_answers, incorrect_answers, total_response_time, formatted_time, grade_percentage
        total_response_time = 0
        formatted_time = "0:00"
        grade_percentage = 0 
        correct_answers.clear()
        incorrect_answers.clear()

    render_current_question(progress_value)

def navigate_to_levels(e, user, module_id):
    try:
        page = e.page
        page.session.set("modules", user.modules)
        page.session.set("module_id", str(module_id))
        if page.session.get("updated_data") is not None:
            page.session.remove("updated_data")
        page.go("/levels")
    except Exception as ex:
        print(f"Error navigating to levels: {ex}")
        page.open(ft.SnackBar(ft.Text("Module navigation error"), bgcolor="#FF0000"))
        page.update()
