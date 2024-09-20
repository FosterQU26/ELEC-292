USE employees;
-- Question 1
select * from employees;
select * from salaries;
-- Question 2
select * from salaries where salary * 1.7 > 100000;
-- Question 3
select AVG(salary) AS AverageSalary FROM salaries WHERE emp_no > 1510;
-- Question 4
select emp_no, avg(salary) as AverageSalary from salaries group by emp_no;
-- Question 5
SELECT employees.first_name, employees.last_name, salaries.salary
FROM employees
INNER JOIN salaries ON employees.emp_no = salaries.emp_no;
-- Question 6
select emp_no, avg(salary) as AverageSalary from salaries group by emp_no;
DROP PROCEDURE IF EXISTS emp_avg_salary;
DELIMITER $$
CREATE PROCEDURE emp_avg_salary(IN p_emp_number INT)
BEGIN
    SELECT AVG(salary)
    FROM salaries
    WHERE emp_no = p_emp_number;
END $$
DELIMITER ;
CALL emp_avg_salary(11300);
