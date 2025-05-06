import flet as ft
import json
import os

def get_module_data(page):
    modules = page.session.get("modules")
    module_id = page.session.get("module_id")

    if not modules:
        try:
            with open("temp_modules.json", "r") as f:
                modules = json.load(f)
        except FileNotFoundError:
            print("Temp module cache not found.")
            return None

    if not module_id:
        print("Module ID not found in session.")
        return None

    for module in modules:
        if str(module.id) == str(module_id):
            return module.levels, module.desc, module.waray_name

    print(f"No matching module with ID {module_id} found.")
    return None

def get_updated_data(page):
    updated_data = page.session.get("updated_data")
    if not updated_data:
        return None

    grade_percentage, formatted_time, correct_answers, incorrect_answers, updated_questions = updated_data
    return {
        "grade": grade_percentage,
        "time": formatted_time,
        "correct": correct_answers,
        "incorrect": incorrect_answers,
        "questions": updated_questions
    }

def clear_temp_module_cache():
    if os.path.exists("temp_modules.json"):
        os.remove("temp_modules.json")

def levels_page(page: ft.Page):
    """Levels selection page"""
    selected_module_levels, selected_module_desc, selected_module_name = get_module_data(page)
    page.title = f"Arami - \"{selected_module_name}\" Levels"
    page.bgcolor = "#FFFFFF"
    page.padding = 0
    level_rows = []
    row = []

    # --- Reset all levels to not completed ---
    for level in selected_module_levels:
        level.completed = False

    updated = page.session.get("updated_data")
    if updated is not None:
        incorrect_answers = updated["incorrect_answers"]
        correct_answers = updated["correct_answers"]
        if updated and updated["questions"]:
            # Get any question to extract identifying info (safe if list isn't empty)
            first_question = updated["questions"][0]

            for level in selected_module_levels:
                if level.lesson_id == first_question.lesson_id and level.module_name == first_question.module_name:
                    level.questions_answers = updated["questions"]
                    # Only set completed if grade is sufficient and this is the just-finished level
                    if updated["grade"] >= 50:
                        level.completed = True
                    else:
                        level.completed = False
                    break
    else:
        incorrect_answers = {}

    def go_back(e):
        """Navigate back to the main menu"""
        selected_module_levels = page.session.get("module_levels")
        modules = page.session.get("modules")
        module_id = page.session.get("module_id")

        if modules and selected_module_levels and module_id:
            for module in modules:
                if str(module.id) == str(module_id):
                    # Update levels in the matched module
                    updated_levels = []
                    for lvl in selected_module_levels:
                        if getattr(lvl, "type", "") != "chaptertest":  # skip chaptertest
                            updated_levels.append(lvl)
                    module.levels = updated_levels
                    break

            # Save updated modules back to the user
            user = page.session.get("user")
            if user:
                user.modules = modules
                page.session.set("user", user)
                page.session.set("incorrect_answers", incorrect_answers)
                page.session.set("correct_answers", correct_answers)

        clear_temp_module_cache()
        page.go("/main-menu")

    def level_select(e, level_num):
        """Handles level selection and navigates to the lesson page."""
        level_index = int(level_num) - 1
        level_data = selected_module_levels[level_index]

        # Check all previous levels (prerequisites)
        prerequisite_levels = selected_module_levels[:level_index]
        all_prerequisites_completed = all(level.completed for level in prerequisite_levels)

        if not all_prerequisites_completed:
            page.open(ft.SnackBar(ft.Text("You must complete previous levels first."), bgcolor="#FF0000"))
            page.update()
            return

        # Optional: prevent re-entering a finished level
        if level_data.completed:
            page.open(ft.SnackBar(ft.Text("Level already completed!"), bgcolor="#FF0000"))
            page.update()
            return

        # Proceed to the lesson
        page.session.set("level_data", level_data)
        print(f"Selected level {level_num}")
        page.go("/lesson")

    def create_level_button(level_number, color="#4285F4"):
        return ft.Container(
            content=ft.Text(
                str(level_number),
                color="#FFFFFF",
                size=20,
                weight=ft.FontWeight.BOLD,
                text_align=ft.TextAlign.CENTER,
            ),
            bgcolor=color,
            width=60,
            height=60,
            border_radius=10,
            alignment=ft.alignment.center,
            data=level_number,
            on_click=lambda e: level_select(page, level_number)
        )

    for index, level in enumerate(selected_module_levels, start=1):
        level_button = create_level_button(index)

        row.append(level_button)

        # When we have 3 buttons, or it's the last button, add the row
        if len(row) == 3 or index == len(selected_module_levels):
            level_rows.append(
                ft.Row(
                    row,
                    alignment=ft.MainAxisAlignment.CENTER,
                    spacing=10
                )
            )
            row = []  # reset for the next row

    # Header with gradient background and title
    header = ft.Container(
        content=ft.Stack([
            ft.Container(
                gradient=ft.LinearGradient(
                    begin=ft.alignment.top_left,
                    end=ft.alignment.bottom_right,
                    colors=["#0066FF", "#9370DB"],
                ),
                height=130,
            ),
            ft.Container(
                content=ft.Text(
                    selected_module_name,
                    size=20,
                    color="#FFFFFF",
                    weight=ft.FontWeight.BOLD,
                    text_align=ft.TextAlign.CENTER,
                ),
                bgcolor="#4285F4",
                border_radius=20,
                padding=ft.padding.symmetric(vertical=8, horizontal=16),
                margin=ft.margin.only(top=100),
                width=240,
                alignment=ft.alignment.center,
            ),
        ]),
        height=130,
    )

    explanation = ft.Container(
        content=ft.Text(
            selected_module_desc,
            color="#000000",
            size=14,
            text_align=ft.TextAlign.CENTER,
        ),
        margin=ft.margin.symmetric(vertical=15, horizontal=20),
        alignment=ft.alignment.center,
    )

    level_grid = ft.Container(
        content=ft.Column(
            level_rows,
            spacing=10, 
        alignment=ft.MainAxisAlignment.CENTER),
        margin=ft.margin.only(top=10),
    )

    back_button = ft.Container(
        content=ft.ElevatedButton(
            text="Back",
            on_click=go_back,
            bgcolor="#4285F4",
            color="#FFFFFF",
            width=100,
            height=40,
        ),
        alignment=ft.alignment.center,
        margin=ft.margin.only(bottom=10),
    )

    bottom_nav = ft.Container(
        content=ft.Row(
            [
                ft.IconButton(icon=ft.Icons.MILITARY_TECH_OUTLINED, icon_color="#FFFFFF", icon_size=24),
                ft.IconButton(icon=ft.Icons.MENU_BOOK_OUTLINED, icon_color="#FFFFFF", icon_size=24),
                ft.IconButton(icon=ft.Icons.HOME_OUTLINED, icon_color="#FFFFFF", icon_size=24),
                ft.IconButton(icon=ft.Icons.PERSON_OUTLINED, icon_color="#FFFFFF", icon_size=24),
                ft.IconButton(icon=ft.Icons.SETTINGS_OUTLINED, icon_color="#FFFFFF", icon_size=24),
            ],
            alignment=ft.MainAxisAlignment.SPACE_AROUND,
        ),
        bgcolor="#280082",
        height=60,
        padding=10
    )

    content = ft.Column(
        [
            header,
            back_button,
            explanation,
            level_grid,
            ft.Container(expand=True),
            bottom_nav,
        ],
        spacing=0,
        expand=True,
    )

    page.views.append(ft.View(
        "/levels",
        controls=[content],
        bgcolor="#FFFFFF",
        padding=0
    ))
    page.update()
