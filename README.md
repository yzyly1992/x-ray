## Get Started!

- Create a venv: python -m venv venv (or python3 -m venv venv on Mac / Linux)
- Activate the venv
  - .\venv\Scripts\activate.ps1 for PowerShell
  - .\venv\Scripts\activate.bat for CMD
  - source venv/bin/activate for Mac / Linux
- Install the dependencies: pip install -r requirements.txt
- Migrate the database: python manage.py migrate
- Load fixture python .\manage.py loaddata users/fixtures/user_and_groups (or python3 ./manage.py loaddata users/fixtures/user_and_groups
on Mac)
- Run the development server: python manage.py runserver

## Note:

- If importing new modules and/or libraries please add them in the requirements.txt file with the correct version
