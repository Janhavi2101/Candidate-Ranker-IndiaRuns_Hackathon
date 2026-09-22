from backend.preprocessing.sample import parse_job_profile

job = parse_job_profile("data/job_description.docx")

print(job)