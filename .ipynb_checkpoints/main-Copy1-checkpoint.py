# import streamlit as st
# from streamlit_multipage import MultiPage


# def input_page(st, **state):
#     st.title("Body Mass Index")

#     weight_ = state["weight"] if "weight" in state else 0.0
#     weight = st.number_input("Your weight (Kg): ", value=weight_)

#     height_ = state["height"] if "height" in state else 0.0
#     height = st.number_input("Your height (m): ", value=height_)

#     if height and weight:
#         MultiPage.save({"weight": weight, "height": height})


# def compute_page(st, **state):
#     st.title("Body Mass Index")

#     if "weight" not in state or "height" not in state:
#         st.warning("Enter your data before computing. Go to the Input Page")
#         return

#     weight = state["weight"]
#     height = state["height"]

#     st.metric("BMI", round(weight / height ** 2, 2))


# app = MultiPage()
# app.st = st

# app.add_app("Input Page", input_page)
# app.add_app("BMI Result", compute_page)

# app.run()

# import streamlit as st
# from streamlit_multipage import MultiPage


# def my_page(st, **state):
#     st.title("My Amazing App")
#     name = st.text_input("Your Name: ")
#     st.write(f"Hello {name}!")


# app = MultiPage()
# app.st = st

# app.add_app("Hello World", my_page)

# app.run()

import streamlit as st
from streamlit_multipage import MultiPage


def input_page(st, **state):
    namespace = "input"
    variables = state[namespace] if namespace in state else {}
    st.title("Tax Deduction")

    salary_ = variables["salary"] if "salary" in variables else 0.0
    salary = st.number_input("Your salary (USD): ", value=salary_)

    tax_percent_ = variables["tax_percent"] if "tax_percent" in variables else 0.0
    tax_percent = st.number_input("Taxes (%): ", value=tax_percent_)

    total = salary * (1 - tax_percent)

    if tax_percent and salary:
        MultiPage.save({"salary": salary, "tax_percent": tax_percent}, namespaces=[namespace])

    if total:
        MultiPage.save({"total": total}, namespaces=[namespace, "result"])


def compute_page(st, **state):
    namespace = "result"
    variables = state[namespace] if namespace in state else {}
    st.title("Your Salary After Taxes")

    if "total" not in variables:
        st.warning("Enter your data before computing. Go to the Input Page")
        return

    total = variables["total"]

    st.metric("Total", round(total, 2))


app = MultiPage()
app.st = st

app.add_app("Input Page", input_page)
app.add_app("Net Salary", compute_page)

app.run()