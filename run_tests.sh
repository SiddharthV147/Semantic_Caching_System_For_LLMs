python -m tests.run_tests   # runs teardown + rebuild, OR:
python -c "
from src.database.db_setup import teardown_all_data, setup_all_databases
teardown_all_data()
setup_all_databases(initial_course_tags=['CS101','MATH202','PHYS404'])
"
