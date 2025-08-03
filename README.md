## Get Started!

- Install dependencies: `uv sync`
- **Install MMDetection (for advanced AI model):**
  ```bash
  ./setup_mmdetection.sh
  ```
  Or manually:
  ```bash
  uv add openmim torch torchvision
  uv run mim install mmengine mmcv mmdet
  ```
- Migrate the database: `uv run python manage.py migrate`
- Load fixture: `uv run python manage.py loaddata users/fixtures/user_and_groups`
- Run the development server: `uv run python manage.py runserver`

## Get Started! (Legacy - pip/venv)

- Create a venv: python -m venv venv (or python3 -m venv venv on Mac / Linux)
- Activate the venv
  - .\venv\Scripts\activate.ps1 for PowerShell
  - .\venv\Scripts\activate.bat for CMD
  - source venv/bin/activate for Mac / Linux
- Install the dependencies: pip install -r requirements.txt
- **Install MMDetection (for advanced AI model):**
  ```bash
  pip install openmim
  mim install mmengine mmcv mmdet
  ```
- Migrate the database: python manage.py migrate
- Load fixture python .\manage.py loaddata users/fixtures/user_and_groups (or python3 ./manage.py loaddata users/fixtures/user_and_groups
on Mac)
- Run the development server: python manage.py runserver

## Note:

- If importing new modules and/or libraries please add them in the requirements.txt file with the correct version
- The application now supports both DenseNet (fallback) and MMDetection DINO models for X-ray analysis
- MMDetection provides more detailed bone disease detection with specific disease classification
