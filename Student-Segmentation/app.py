
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="StudentSeg AI",
    page_icon="🎓",
    layout="wide"
)

FEATURES = ["marks", "attendance", "study_hours"]

LABELS = {
    "marks": "Marks (%)",
    "attendance": "Attendance (%)",
    "study_hours": "Study hours/day"
}


# =====================================================
# 2. GENERATE SAMPLE STUDENT DATA
# =====================================================

def make_sample_data(n=180, seed=42):
    rng = np.random.default_rng(seed)

    group_sizes = [
        int(n * 0.38),
        int(n * 0.37)
    ]
    group_sizes.append(n - sum(group_sizes))

    groups = [
        (84, 7, 91, 6, 4.2, 0.8),
        (67, 8, 79, 9, 2.5, 0.7),
        (48, 10, 65, 12, 1.2, 0.6)
    ]

    rows = []
    student_number = 1

    for count, params in zip(group_sizes, groups):
        marks_mean, marks_sd, att_mean, att_sd, hours_mean, hours_sd = params

        for _ in range(count):
            rows.append({
                "student_id": f"STU{student_number:03d}",
                "marks": float(np.clip(
                    rng.normal(marks_mean, marks_sd), 0, 100
                )),
                "attendance": float(np.clip(
                    rng.normal(att_mean, att_sd), 0, 100
                )),
                "study_hours": float(np.clip(
                    rng.normal(hours_mean, hours_sd), 0, 12
                ))
            })
            student_number += 1

    return pd.DataFrame(rows)


# =====================================================
# 3. CLEAN AND VALIDATE DATA
# =====================================================

def clean_column_names(data):
    data = data.copy()

    data.columns = [
        str(column).strip().lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        for column in data.columns
    ]

    aliases = {
        "student": "student_id",
        "studentid": "student_id",
        "id": "student_id",
        "name": "student_name",
        "marks_%": "marks",
        "mark": "marks",
        "score": "marks",
        "grade": "marks",
        "marks_percent": "marks",
        "attendance_%": "attendance",
        "attendance_percent": "attendance",
        "attendance_percentage": "attendance",
        "study_hours_per_day": "study_hours",
        "studyhours": "study_hours",
        "hours": "study_hours"
    }

    data = data.rename(
        columns={column: aliases.get(column, column)
                 for column in data.columns}
    )

    return data


def prepare_data(raw_data):
    data = clean_column_names(raw_data)

    missing = [column for column in FEATURES
               if column not in data.columns]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Required: marks, attendance, study_hours."
        )

    if "student_id" not in data.columns:
        data["student_id"] = [
            f"STU{i:03d}" for i in range(1, len(data) + 1)
        ]

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(subset=FEATURES).copy()

    if data.empty:
        raise ValueError("No valid numeric student records were found.")

    if len(data) < 2:
        raise ValueError("At least two valid students are required.")

    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Student data contains infinite or invalid values.")

    if not data["marks"].between(0, 100).all():
        raise ValueError("Marks must be between 0 and 100.")

    if not data["attendance"].between(0, 100).all():
        raise ValueError("Attendance must be between 0 and 100.")

    if not data["study_hours"].between(0, 24).all():
        raise ValueError("Study hours must be between 0 and 24.")

    data["student_id"] = data["student_id"].astype(str)

    return data.reset_index(drop=True)


# =====================================================
# 4. INTERPRET CLUSTER PROFILES
# =====================================================

def describe_cluster(row, overall):
    if (
        row["marks"] >= overall["marks"] + 7
        and row["attendance"] >= overall["attendance"]
    ):
        return "Higher academic performance"

    if (
        row["marks"] <= overall["marks"] - 7
        and row["attendance"] < overall["attendance"]
    ):
        return "May benefit from extra support"

    if (
        row["study_hours"] >= overall["study_hours"] + 0.7
        and row["marks"] < overall["marks"]
    ):
        return "High effort, developing results"

    if row["marks"] >= overall["marks"]:
        return "Above-average performance"

    return "Developing academic pattern"


# =====================================================
# 5. CUSTOM DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #12243a, #2563eb);
    color: white;
    margin-bottom: 20px;
}

.hero h1, .hero p {
    color: white !important;
}

div[data-testid="stMetric"] {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #334155;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #f8fafc !important;
}

h1, h2, h3 {
    color: #f8fafc;
}

div[data-testid="stForm"] {
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 14px;
}

</style>

<div class="hero">
    <h1>🎓 StudentSegmentation</h1>
    <p>
        Student Segmentation using Machine Learning
    </p>
    <p>
        Analyze academic behavior using K-Means clustering.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 6. SIDEBAR SETTINGS
# =====================================================

with st.sidebar:
    st.header("⚙️ Model Settings")

    data_source = st.radio(
        "Choose data source",
        ["Use sample data", "Upload my CSV"]
    )

    uploaded_file = None

    if data_source == "Upload my CSV":
        uploaded_file = st.file_uploader(
            "Upload student dataset",
            type=["csv"]
        )

        st.caption(
            "Required CSV columns: marks, attendance, study_hours. "
            "student_id is optional."
        )

    sample_size = st.slider(
        "Number of sample students",
        min_value=60,
        max_value=600,
        value=180,
        step=20,
        disabled=(data_source != "Use sample data")
    )

    k = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=6,
        value=3
    )

    st.divider()

    st.caption("Algorithm: K-Means")
    st.caption("Learning: Unsupervised Machine Learning")


# =====================================================
# 7. LOAD DATA AND TRAIN MODEL
# =====================================================

try:
    if data_source == "Use sample data":
        raw_df = make_sample_data(sample_size)
        source_label = "Illustrative sample data"

    elif uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        source_label = uploaded_file.name

    else:
        st.info("Upload a CSV file using the sidebar to continue.")
        st.stop()

    df = prepare_data(raw_df)

    if len(df) < k:
        st.error(
            f"You selected K={k}, but only {len(df)} valid records exist."
        )
        st.stop()

    # Scale the features before clustering.
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[FEATURES])

    # Train K-Means on the selected dataset.
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(scaled_features)

    # Name groups according to their average marks.
    cluster_means_raw = df.groupby("cluster")[FEATURES].mean()
    ordered_clusters = cluster_means_raw.sort_values("marks").index.tolist()

    name_map = {
        cluster_id: f"Group {position + 1}"
        for position, cluster_id in enumerate(ordered_clusters)
    }

    df["cluster_name"] = df["cluster"].map(name_map)

    cluster_means = (
        df.groupby("cluster_name")[FEATURES]
        .mean()
        .reindex([name_map[c] for c in ordered_clusters])
    )

    overall = df[FEATURES].mean()

    st.caption(
        f"Data source: {source_label} | "
        f"Valid students: {len(df)} | "
        f"Features: marks, attendance, study hours"
    )


    # =================================================
    # 8. DASHBOARD METRICS
    # =================================================

    st.header("📊 Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Students Analyzed", f"{len(df):,}")
    c2.metric("Average Marks", f"{df['marks'].mean():.1f}%")
    c3.metric("Average Attendance", f"{df['attendance'].mean():.1f}%")
    c4.metric("Clusters Created", str(k))

    st.divider()


    # =================================================
    # 9. CLUSTER VISUALIZATION
    # =================================================

    st.header("📈 Student Cluster Visualization")

    left, right = st.columns([1.6, 1])

    with left:
        fig = px.scatter(
            df,
            x="study_hours",
            y="marks",
            color="cluster_name",
            size="attendance",
            hover_data=[
                "student_id",
                "marks",
                "attendance",
                "study_hours"
            ],
            labels={
                "study_hours": "Study Hours per Day",
                "marks": "Marks (%)",
                "cluster_name": "Student Group",
                "attendance": "Attendance (%)"
            },
            title="Students Grouped by Study Hours and Marks",
            template="plotly_dark"
        )

        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        counts = (
            df["cluster_name"]
            .value_counts()
            .rename_axis("Student Group")
            .reset_index(name="Students")
        )

        fig_count = px.bar(
            counts,
            x="Student Group",
            y="Students",
            color="Student Group",
            text="Students",
            title="Students in Each Group",
            template="plotly_dark"
        )

        fig_count.update_layout(
            showlegend=False,
            height=430
        )

        st.plotly_chart(fig_count, use_container_width=True)


    # =================================================
    # 10. CLUSTER PROFILES
    # =================================================

    st.header("🧭 Cluster Profiles")

    profile = cluster_means.copy().reset_index()

    profile = profile.rename(
        columns={"cluster_name": "Student Group"}
    )

    group_counts = df["cluster_name"].value_counts()

    profile["Number of Students"] = (
        profile["Student Group"].map(group_counts)
    )

    profile["Interpretation"] = profile.apply(
        lambda row: describe_cluster(
            row.rename({
                "Marks (%)": "marks"
            }) if False else pd.Series({
                "marks": row["marks"],
                "attendance": row["attendance"],
                "study_hours": row["study_hours"]
            }),
            overall
        ),
        axis=1
    )

    profile = profile.rename(columns=LABELS)

    for column in [
        "Marks (%)",
        "Attendance (%)",
        "Study hours/day"
    ]:
        profile[column] = profile[column].round(1)

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Group descriptions are summaries of cluster averages, "
        "not official academic judgments."
    )

    st.divider()


    # =================================================
    # 11. INDIVIDUAL STUDENT INPUT AND ANALYSIS
    # =================================================

    st.header("🧑‍🎓 Analyze an Individual Student")

    st.write(
        "Enter a student's details below to see their profile, "
        "predicted group, and comparison with that group's averages."
    )

    with st.form("individual_student_form"):

        student_id_input = st.text_input(
            "Student ID",
            placeholder="Example: STU201"
        )

        student_name_input = st.text_input(
            "Student Name",
            placeholder="Enter student's full name"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            student_marks = st.number_input(
                "Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=1.0
            )

        with col2:
            student_attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0
            )

        with col3:
            student_hours = st.number_input(
                "Study Hours per Day",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        analyze_student = st.form_submit_button(
            "🔍 Analyze Student",
            use_container_width=True
        )

    if analyze_student:

        if (
            not student_id_input.strip()
            or not student_name_input.strip()
        ):
            st.error(
                "Please enter both Student ID and Student Name."
            )

        else:
            # Prepare the new student's features.
            new_student = pd.DataFrame(
                [[student_marks, student_attendance, student_hours]],
                columns=FEATURES
            )

            # Use the SAME scaler and model trained above.
            scaled_student = scaler.transform(new_student)

            predicted_cluster = int(
                model.predict(scaled_student)[0]
            )

            predicted_group = name_map[predicted_cluster]

            st.success("Student analysis completed!")

            st.subheader(
                f"👤 Student Profile: {student_name_input.strip()}"
            )

            st.write(f"**Student ID:** {student_id_input.strip()}")

            m1, m2, m3 = st.columns(3)

            m1.metric("Marks", f"{student_marks:.1f}%")
            m2.metric("Attendance", f"{student_attendance:.1f}%")
            m3.metric("Study Hours", f"{student_hours:.1f} hrs/day")

            st.subheader("🤖 K-Means Segmentation Result")

            st.metric("Assigned Group", predicted_group)

            st.info(
                f"The student has been assigned to **{predicted_group}** "
                "based on marks, attendance, and study hours."
            )

            # Find the average features of the assigned group.
            group_average = df[
                df["cluster_name"] == predicted_group
            ][FEATURES].mean()

            st.subheader("📊 Student vs. Group Average")

            comparison = pd.DataFrame({
                "Feature": [
                    "Marks (%)",
                    "Attendance (%)",
                    "Study Hours/day"
                ],
                "Entered Student": [
                    student_marks,
                    student_attendance,
                    student_hours
                ],
                "Group Average": [
                    group_average["marks"],
                    group_average["attendance"],
                    group_average["study_hours"]
                ]
            })

            comparison["Entered Student"] = comparison[
                "Entered Student"
            ].round(2)

            comparison["Group Average"] = comparison[
                "Group Average"
            ].round(2)

            fig_compare = px.bar(
                comparison,
                x="Feature",
                y=["Entered Student", "Group Average"],
                barmode="group",
                title="Individual Student vs. Assigned Group",
                template="plotly_dark"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True
            )

            # Create a downloadable individual report.
            individual_result = pd.DataFrame([{
                "student_id": student_id_input.strip(),
                "student_name": student_name_input.strip(),
                "marks": student_marks,
                "attendance": student_attendance,
                "study_hours": student_hours,
                "predicted_cluster": predicted_cluster,
                "assigned_group": predicted_group
            }])

            st.download_button(
                "⬇️ Download Individual Student Report",
                data=individual_result.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name=(
                    f"{student_id_input.strip()}_result.csv"
                ),
                mime="text/csv"
            )

    st.divider()


    # =================================================
    # 12. SEARCH EXISTING STUDENTS
    # =================================================

    st.header("🔎 Search Existing Student Results")

    search = st.text_input(
        "Search student ID",
        placeholder="Example: STU001"
    )

    display_df = df.copy()

    if search.strip():
        display_df = display_df[
            display_df["student_id"].str.contains(
                search.strip(),
                case=False,
                na=False,
                regex=False
            )
        ]

    display_df = display_df[
        [
            "student_id",
            "marks",
            "attendance",
            "study_hours",
            "cluster_name"
        ]
    ].rename(columns={
        "student_id": "Student ID",
        "marks": "Marks (%)",
        "attendance": "Attendance (%)",
        "study_hours": "Study hours/day",
        "cluster_name": "Assigned Group"
    })

    st.dataframe(
        display_df.round({
            "Marks (%)": 1,
            "Attendance (%)": 1,
            "Study hours/day": 1
        }),
        use_container_width=True,
        hide_index=True
    )

    if display_df.empty:
        st.warning("No matching student was found.")

    st.download_button(
        "⬇️ Download All Clustered Results",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="studentseg_results.csv",
        mime="text/csv"
    )


    # =================================================
    # 13. EXPLAIN THE ML PIPELINE
    # =================================================

    with st.expander("ℹ️ How the Machine Learning Model Works"):

        st.markdown("""
        1. **Feature selection:** marks, attendance, and study hours.
        2. **Data preprocessing:** invalid and missing feature rows are removed.
        3. **Feature scaling:** StandardScaler puts features on a comparable scale.
        4. **K-Means clustering:** groups similar student records.
        5. **Individual prediction:** a new student's features are transformed
           using the fitted scaler and assigned to the nearest learned cluster.
        6. **Interpretation:** group averages help describe the discovered patterns.

        K-Means is an **unsupervised machine learning algorithm**.
        Cluster assignments are not the same as predicting exam results.
        
        The built-in dataset is synthetic demonstration data. Use a suitable
        real dataset and validate the analysis before drawing research conclusions.
        """)

except (
    ValueError,
    pd.errors.ParserError,
    UnicodeDecodeError
) as exc:
    st.error(str(exc))
    st.info(
        "Check your dataset and ensure it contains numeric "
        "marks, attendance, and study_hours columns."
    )

except Exception as exc:
    st.error(f"An unexpected error occurred: {exc}")
    st.caption(
        "Check that all required libraries are installed."
    )
