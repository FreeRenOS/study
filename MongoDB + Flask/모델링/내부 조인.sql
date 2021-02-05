
select e.sabun, e.name, d.name from employee e
	inner join dept d
	on e.dept_id = d.dept_id;
    