# Python Fundamentals for Data Analysis: Project Documentation

## 1. Project Overview
This project demonstrates the application of Python programming basics in the context of Telecommunications and E-commerce data. The goal is to show how raw data is stored, processed, and categorized using standard programming structures. These steps are essential for the **Data Preprocessing** stage in Machine Learning.

## 2. Variables and Data Types
The script initializes several variables to represent customer profiles. Choosing the correct data type is the first step in data organization:

* **Integers (`int`):** Used for whole numbers that do not require decimals, such as `customer_age`.
* **Floats (`float`):** Used for continuous values like `data_usage_gb` or financial amounts (e.g., monthly bills).
* **Booleans (`bool`):** Used for binary logic (`True`/`False`), such as checking if a customer `has_contract`.
* **Lists (`list`):** Used to store a collection of items in a single variable, like a 6-month history of `bills`.

## 3. Data Structures: Dictionaries
A dictionary structure is implemented to represent a product in an e-commerce store. 
* **Key-Value Pairs:** This format allows the code to link attributes (like "name" or "price") directly to their specific values.
* **Dynamic Updates:** The code shows how to add new information, such as a `discount` key, to an existing record without changing the rest of the data structure.

## 4. Logic and Functions
Two main functions are used to automate data analysis:

### Customer Classification (`classify_customer`)
This function uses `if-elif-else` conditional logic to sort customers into three tiers based on their spending:
1.  **Premium:** Spend > 150,000
2.  **Standard:** Spend between 50,000 and 150,000
3.  **Basic:** Spend < 50,000

### Churn Rate Calculation (`calculate_churn_rate`)
This function calculates the percentage of customers who leave a service. It takes the total number of customers and the number of churned customers as inputs to return a percentage value.

## 5. Loops and Iteration
The program uses `for` loops to handle lists of data efficiently:
* **Enumeration:** The `enumerate()` function is used to track both the index (position) and the value of items in a list simultaneously.
* **Manual Aggregation:** A loop is used to manually calculate the `total` and `average` of a list. This demonstrates the algorithmic logic of how computers sum and divide data points without using built-in shortcuts.

## 6. Final Data Analysis Execution
In the final part of the script, all the concepts above are combined to analyze a dataset of 8 customers:
1.  The program **iterates** through the list of customer dictionaries.
2.  It **classifies** each customer using the logic defined in the functions.
3.  It **counts** the number of customers in each category using counter variables.
4.  Finally, it **prints** a summary report showing the distribution of customers (e.g., Premium: 3, Standard: 3, Basic: 2).

---

### Conclusion
The code successfully demonstrates how to manage data types, create logical functions, and use loops for data reporting. These are the core skills required for preparing datasets for Artificial Intelligence and Machine Learning models.