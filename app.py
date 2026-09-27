import streamlit as st
import json
import os
import html
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="College AI Chatbot",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD COLLEGE DATA
# =========================================================

DATA_FILE = os.path.join("data", "college_data.json")

try:
    with open(DATA_FILE, "r", encoding="utf-8") as file:
        college_data = json.load(file)
except FileNotFoundError:
    st.error("college_data.json file not found. Please check the data folder.")
    st.stop()
except json.JSONDecodeError:
    st.error("college_data.json contains invalid JSON.")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# CSS / UI DESIGN
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f8fbff 0%,
            #eef5ff 50%,
            #ffffff 100%
        );
    }

    .college-logo {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: linear-gradient(135deg, #4f46e5, #2563eb);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 35px;
        margin: 0 auto 10px auto;
        box-shadow: 0 5px 20px rgba(37, 99, 235, 0.25);
    }

    .college-name {
        text-align: center;
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .main-title {
        text-align: center;
        font-size: 25px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        margin-bottom: 10px;
    }

    .online-status {
        text-align: center;
        font-size: 14px;
        margin-bottom: 25px;
    }

    .welcome-card {
        padding: 20px;
        border-radius: 15px;
        background: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="college-logo">🎓</div>',
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="college-name">
        {html.escape(college_data["college_name"])}
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        AI-Powered College Information Chatbot
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        🤖 Your Smart Virtual Assistant for College Information
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="online-status">
        🟢 AI Assistant Online
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# WELCOME MESSAGE
# =========================================================

if len(st.session_state.messages) == 0:
    st.markdown(
        f"""
        <div class="welcome-card">
            <h3>👋 Welcome!</h3>
            <p>
                I am your AI-powered college assistant.
                You can ask me about the college, courses,
                timings, hostel, library, transportation and facilities.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# QUESTION - ANSWER KNOWLEDGE BASE
# =========================================================

qa_data = [

    # College
    {
        "questions": [
            "what is the college name",
            "tell me the college name",
            "name of college",
            "which college is this",
            "college name"
        ],
        "answer": f"The college name is **{college_data['college_name']}**."
    },

    {
        "questions": [
            "where is the college located",
            "college location",
            "where is college",
            "college address",
            "location of college"
        ],
        "answer": f"The college is located at **{college_data['college_location']}**."
    },

    # Timings
    {
        "questions": [
            "what are college timings",
            "college timing",
            "college working hours",
            "when does college start",
            "when does college close",
            "college hours",
            "working time of college"
        ],
        "answer": (
            f"The college has two shifts:\n\n"
            f"🕗 **First Shift:** {college_data['college_timings']['first_shift']}\n\n"
            f"🕙 **Second Shift:** {college_data['college_timings']['second_shift']}"
        )
    },

    {
        "questions": [
            "first shift timing",
            "first shift time",
            "first shift",
            "morning shift",
            "what time first shift starts",
            "what time first shift ends"
        ],
        "answer": (
            f"The **First Shift** timing is "
            f"**{college_data['college_timings']['first_shift']}**."
        )
    },

    {
        "questions": [
            "second shift timing",
            "second shift time",
            "second shift",
            "what time second shift starts",
            "what time second shift ends"
        ],
        "answer": (
            f"The **Second Shift** timing is "
            f"**{college_data['college_timings']['second_shift']}**."
        )
    },

    # Courses
    {
        "questions": [
            "what courses are available",
            "what branches are available",
            "courses offered",
            "branches offered",
            "what departments are available",
            "available courses",
            "available branches"
        ],
        "answer": (
            "The available courses/branches are:\n\n"
            + "\n".join(
                [f"• {course}" for course in college_data["courses"]]
            )
        )
    },

    {
        "questions": [
            "is cse available",
            "does college have cse",
            "computer science course",
            "computer science engineering",
            "cse branch"
        ],
        "answer": "Yes, **Computer Science and Engineering (CSE)** is available."
    },

    # CSE Specializations
    {
        "questions": [
            "what are cse specializations",
            "cse sub branches",
            "cse specializations",
            "cse sub branches available",
            "specializations in cse",
            "branches under cse",
            "what are the cse branches",
            "cse options"
        ],
        "answer": (
            "The CSE sub-branches/specializations are:\n\n"
            "• **Artificial Intelligence and Machine Learning (AI & ML)**\n"
            "• **Artificial Intelligence and Data Science (AI & DS)**\n"
            "• **Data Science (DS)**"
        )
    },

    {
        "questions": [
            "is ai ml available",
            "ai and ml branch",
            "artificial intelligence machine learning",
            "ai ml specialization"
        ],
        "answer": (
            "Yes. **Artificial Intelligence and Machine Learning (AI & ML)** "
            "is available under CSE."
        )
    },

    {
        "questions": [
            "is ai ds available",
            "ai and data science",
            "artificial intelligence and data science",
            "ai ds specialization"
        ],
        "answer": (
            "Yes. **Artificial Intelligence and Data Science (AI & DS)** "
            "is available under CSE."
        )
    },

    # Hostel
    {
        "questions": [
            "is hostel available",
            "does college have hostel",
            "is there hostel",
            "hostel available",
            "do you provide hostel",
            "does college provide hostel",
            "hostel facility"
        ],
        "answer": (
            "Yes, **Boys Hostel** is available."
        )
    },

    {
        "questions": [
            "which hostel is available",
            "what hostel is available",
            "hostel type",
            "boys hostel",
            "hostel for boys"
        ],
        "answer": (
            "The available hostel facility is **Boys Hostel**."
        )
    },

    # Library
    {
        "questions": [
            "library timing",
            "library timings",
            "what are library timings",
            "library hours",
            "when is library open",
            "when does library open",
            "when does library close",
            "library time"
        ],
        "answer": (
            f"The library is available from "
            f"**{college_data['library']['timing']}**."
        )
    },

    {
        "questions": [
            "is library available",
            "does college have library",
            "library available",
            "library facility"
        ],
        "answer": (
            f"Yes, the college has a **Library**.\n\n"
            f"🕙 Timing: **{college_data['library']['timing']}**"
        )
    },

    # Transportation
    {
        "questions": [
            "is bus facility available",
            "college bus available",
            "does college provide bus",
            "transportation facility",
            "transport facility",
            "bus facility",
            "college transportation"
        ],
        "answer": (
            "Yes, **College Bus** transportation is available."
        )
    },

    # Facilities
    {
        "questions": [
            "what facilities are available",
            "college facilities",
            "facilities in college",
            "what facilities does college have",
            "available facilities"
        ],
        "answer": (
            "The available facilities include:\n\n"
            "• 📚 **Library**\n"
            "• 🏠 **Boys Hostel**\n"
            "• 🚌 **College Bus**"
        )
    }
]


# =========================================================
# PREPARE TF-IDF MODEL
# =========================================================

question_list = []
answer_list = []

for item in qa_data:
    for question in item["questions"]:
        question_list.append(question.lower())
        answer_list.append(item["answer"])

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

question_vectors = vectorizer.fit_transform(question_list)


# =========================================================
# RESPONSE FUNCTION
# =========================================================

def get_response(user_question):

    question = user_question.lower().strip()

    # -----------------------------------------
    # Empty question
    # -----------------------------------------

    if not question:
        return "Please enter a question."


    # -----------------------------------------
    # Greetings
    # -----------------------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "hai",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if question in greetings:
        return (
            f"Hello! 👋\n\n"
            f"I am the AI assistant for "
            f"**{college_data['college_name']}**.\n\n"
            "You can ask me about courses, timings, hostel, "
            "library, transportation and facilities."
        )


    # -----------------------------------------
    # Who are you?
    # -----------------------------------------

    if (
        "who are you" in question
        or "what are you" in question
        or "what can you do" in question
        or "your purpose" in question
    ):
        return (
            "I am an **AI-Powered College Information Chatbot** 🤖.\n\n"
            "I can provide information about the college, "
            "courses, CSE specializations, timings, hostel, "
            "library, transportation and facilities."
        )


    # -----------------------------------------
    # Direct college name
    # -----------------------------------------

    if (
        "college name" in question
        or "name of college" in question
        or "which college" in question
    ):
        return (
            f"The college name is **{college_data['college_name']}**."
        )


    # -----------------------------------------
    # Location
    # -----------------------------------------

    if (
        "location" in question
        or "where is the college" in question
        or "where is college" in question
        or "address" in question
    ):
        return (
            f"The college is located at "
            f"**{college_data['college_location']}**."
        )


    # -----------------------------------------
    # CSE sub branches
    # IMPORTANT: Check before general courses
    # -----------------------------------------

    cse_keywords = [
        "cse sub",
        "cse special",
        "specialization",
        "specialisations",
        "sub branch",
        "sub branches",
        "branches under cse",
        "branches in cse"
    ]

    if any(keyword in question for keyword in cse_keywords):
        return (
            "The CSE sub-branches/specializations are:\n\n"
            "• **Artificial Intelligence and Machine Learning (AI & ML)**\n"
            "• **Artificial Intelligence and Data Science (AI & DS)**\n"
            "• **Data Science (DS)**"
        )


    # -----------------------------------------
    # AI & ML
    # -----------------------------------------

    if (
        "ai ml" in question
        or "ai and ml" in question
        or "artificial intelligence and machine learning" in question
    ):
        return (
            "Yes. **Artificial Intelligence and Machine Learning (AI & ML)** "
            "is available under CSE."
        )


    # -----------------------------------------
    # AI & DS
    # -----------------------------------------

    if (
        "ai ds" in question
        or "ai and ds" in question
        or "artificial intelligence and data science" in question
    ):
        return (
            "Yes. **Artificial Intelligence and Data Science (AI & DS)** "
            "is available under CSE."
        )


    # -----------------------------------------
    # Hostel
    # -----------------------------------------

    hostel_keywords = [
        "hostel",
        "boys hostel",
        "hostel facility",
        "hostel available",
        "hostel provide"
    ]

    if any(keyword in question for keyword in hostel_keywords):
        return "Yes, **Boys Hostel** is available."


    # -----------------------------------------
    # Library
    # -----------------------------------------

    library_keywords = [
        "library",
        "libary"
    ]

    if any(keyword in question for keyword in library_keywords):

        if (
            "time" in question
            or "timing" in question
            or "hour" in question
            or "open" in question
            or "close" in question
        ):
            return (
                f"The library is available from "
                f"**{college_data['library']['timing']}**."
            )

        return (
            f"Yes, the college has a **Library**.\n\n"
            f"🕙 Timing: **{college_data['library']['timing']}**"
        )


    # -----------------------------------------
    # Transportation
    # -----------------------------------------

    transport_keywords = [
        "bus",
        "transport",
        "transportation"
    ]

    if any(keyword in question for keyword in transport_keywords):
        return "Yes, **College Bus** transportation is available."


    # -----------------------------------------
    # Facilities
    # -----------------------------------------

    if (
        "facilities" in question
        or "facility" in question
    ):
        return (
            "The available facilities include:\n\n"
            "• 📚 **Library**\n"
            "• 🏠 **Boys Hostel**\n"
            "• 🚌 **College Bus**"
        )


    # -----------------------------------------
    # College timings
    # -----------------------------------------

    timing_keywords = [
        "college timing",
        "college timings",
        "college hours",
        "working hours",
        "working time",
        "college time",
        "first shift",
        "second shift"
    ]

    if any(keyword in question for keyword in timing_keywords):

        if "first shift" in question:
            return (
                f"The **First Shift** timing is "
                f"**{college_data['college_timings']['first_shift']}**."
            )

        if "second shift" in question:
            return (
                f"The **Second Shift** timing is "
                f"**{college_data['college_timings']['second_shift']}**."
            )

        return (
            f"The college has two shifts:\n\n"
            f"🕗 **First Shift:** "
            f"{college_data['college_timings']['first_shift']}\n\n"
            f"🕙 **Second Shift:** "
            f"{college_data['college_timings']['second_shift']}"
        )


    # -----------------------------------------
    # General courses / branches
    # -----------------------------------------

    course_keywords = [
        "course",
        "courses",
        "branch",
        "branches",
        "department",
        "departments"
    ]

    if any(keyword in question for keyword in course_keywords):

        return (
            "The available courses/branches are:\n\n"
            + "\n".join(
                [f"• {course}" for course in college_data["courses"]]
            )
        )


    # -----------------------------------------
    # TF-IDF similarity
    # -----------------------------------------

    try:

        user_vector = vectorizer.transform([question])

        similarities = cosine_similarity(
            user_vector,
            question_vectors
        ).flatten()

        best_index = similarities.argmax()
        best_score = similarities[best_index]

        if best_score >= 0.25:
            return answer_list[best_index]

    except Exception:
        pass


    # -----------------------------------------
    # Unknown question
    # -----------------------------------------

    return (
        "Sorry, I don't have information about that yet. 🤔\n\n"
        "You can ask me about:\n\n"
        "• College name\n"
        "• College location\n"
        "• College timings\n"
        "• Courses and branches\n"
        "• CSE specializations\n"
        "• Boys Hostel\n"
        "• Library\n"
        "• College Bus\n"
        "• College facilities"
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 College Assistant")

    st.markdown("---")

    st.markdown("### 💡 You can ask")

    st.markdown(
        """
        - 🏫 College name
        - 📍 College location
        - ⏰ College timings
        - 📚 Courses
        - 🤖 CSE specializations
        - 🏠 Boys Hostel
        - 📖 Library
        - 🚌 College Bus
        - 🏢 Facilities
        """
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    st.markdown("### 📊 Chat Information")

    st.write(
        f"Messages: {len(st.session_state.messages)}"
    )


# =========================================================
# SUGGESTED QUESTIONS
# =========================================================

st.markdown("### 💬 Suggested Questions")

suggested_questions = [
    "What is the college name?",
    "Where is the college located?",
    "What are the college timings?",
    "What courses are available?",
    "What are the CSE specializations?",
    "Is hostel available?",
    "What are the library timings?",
    "Is college bus available?"
]

cols = st.columns(4)

for index, question in enumerate(suggested_questions):

    with cols[index % 4]:

        if st.button(
            question,
            key=f"suggestion_{index}",
            use_container_width=True
        ):

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            response = get_response(question)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response
                }
            )

            st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask me anything about the college..."
)


if user_question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Generate response
    response = get_response(user_question)

    # Add assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    st.rerun()


# =========================================================
# DOWNLOAD CHAT HISTORY
# =========================================================

if len(st.session_state.messages) > 0:

    chat_text = ""

    for message in st.session_state.messages:

        if message["role"] == "user":
            chat_text += f"User: {message['content']}\n\n"

        else:
            chat_text += f"AI Assistant: {message['content']}\n\n"

        chat_text += "-" * 50 + "\n\n"

    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="college_chat_history.txt",
        mime="text/plain"
    )