from fastapi import FastAPI, HTTPException, BackgroundTasks, Form, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import google.generativeai as genai
import json
import re
import uuid
from fastapi import Request
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import inch
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import io

app = FastAPI(title="Case Study Generator API", description="API for generating academic case studies using Gemini AI")
templates = Jinja2Templates(directory="templates")
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "https://writeit-8v58f8ke2-codeboyspaces-projects.vercel.app",
    "https://writeit-8v58f8ke2-codeboyspaces-projects.vercel.app/",
    "http://localhost:8080",
    "http://localhost:5173",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Route to serve the HTML frontend
@app.get("/")
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- Configuration ---
API_KEY = "AIzaSyBPcKhQboNsv7kGCSaWALyfcIgGgwNqvrk"  # Consider using environment variables instead
MODEL_NAME = "gemini-2.0-flash"
FIRSTPAGE_PDF = "template.pdf"  # First page template file

# --- Data Models ---
class UserDetails(BaseModel):
    name: str
    class_name: str
    reg_no: str
    subject_name: str
    subject_code: str
    date_of_submission: Optional[str] = None

class CaseStudyRequest(BaseModel):
    title: str
    subheadings: List[str]
    num_pages: int
    user_details: UserDetails

class JobStatus(BaseModel):
    job_id: str
    status: str
    file_name: Optional[str] = None

# --- Global storage for job status ---
jobs_status = {}

# --- Functions ---
def configure_genai():
    """Configures the Generative AI client."""
    if not API_KEY:
        raise ValueError("API_KEY is not set.")
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        return model
    except Exception as e:
        raise Exception(f"Error configuring Generative AI: {e}")

def extract_json_from_text(text):
    """Extract JSON content from a text that might contain additional formatting or markdown."""
    # Look for content between triple backticks with json
    json_pattern = r"```(?\:json)?(.*?)```"
    match = re.search(json_pattern, text, re.DOTALL)

    if match:
        # Found JSON inside code blocks
        json_text = match.group(1).strip()
    else:
        # Try using the entire text as JSON
        json_text = text.strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        # Try fixing common JSON issues
        try:
            # Sometimes JSON might be incomplete - try to fix it
            if json_text.endswith('...'):
                # Try to fix truncated JSON
                # Add closing braces for any opening ones
                open_braces = json_text.count('{')
                close_braces = json_text.count('}')
                if open_braces > close_braces:
                    json_text = json_text.rstrip('.')
                    json_text += '}' * (open_braces - close_braces)
                    return json.loads(json_text)
        except:
            pass

        # If all attempts fail, return None
        return None

def generate_case_study_content(model, title, subheadings, num_pages):
    """Generates case study content using the Gemini model and returns it in JSON format."""
    # First try to get content as JSON
    try:
        # Construct a detailed prompt for JSON format
        subheading_list_str = "\n".join([f"- {s}" for s in subheadings])
        json_prompt = f"""Please generate a comprehensive case study titled "{title}".

The case study should be structured around the following key sections:
{subheading_list_str}

Provide detailed content for each section, ensuring a logical flow between them. The overall length should be suitable for a document that is approximately {num_pages} pages long when formatted. Focus on clarity, accuracy, and professional tone.

Return the content in JSON format with each subheading as a key and its content as the value. Use valid JSON structure without any markdown formatting around it.
"""

        response = model.generate_content(json_prompt)

        if not response.parts:
            return None

        generated_text = response.text
        content_json = extract_json_from_text(generated_text)

        if content_json:
            return content_json

        # If JSON parsing failed, try a simpler approach with plain text
        content_dict = {}

        # Generate content for each subheading separately
        for subheading in subheadings:
            section_prompt = f"""For a case study titled "{title}", write a detailed section on "{subheading}".

Please provide detailed, professional content suitable for an academic case study. The content should be approximately {num_pages//len(subheadings)} pages when formatted. Focus on clarity, accuracy, and professional tone.

Do not include any JSON formatting or code blocks in your response. Just provide the plain text content for this section.
"""

            try:
                section_response = model.generate_content(section_prompt)
                if section_response.parts:
                    content_dict[subheading] = section_response.text
                else:
                    content_dict[subheading] = f"Content for {subheading} could not be generated."
            except Exception as e:
                content_dict[subheading] = f"Content for {subheading} could not be generated due to an error."

        return content_dict

    except Exception as e:
        raise Exception(f"Error during content generation: {e}")

def convert_markdown_to_html(text):
    """Convert Markdown formatting to HTML for use in ReportLab paragraphs."""
    # Handle bold text: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)

    # Handle italic text: *text* -> <i>text</i>
    text = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', text)

    # Handle links: [text](url) -> <a href="url">text</a>
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)

    # Handle code: `text` -> <code>text</code>
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)

    return text

def process_markdown_text(text):
    """Process markdown text to handle bullet points and formatting for PDF."""
    # Clean up any code block markers
    text = re.sub(r'```(?\:json)?|```', '', text)

    # Split into paragraphs
    paragraphs = []
    current_text = ""

    # Process the text line by line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            if current_text:
                paragraphs.append(current_text)
                current_text = ""
            continue

        # Check for bullet points or headers
        if line.startswith('*') and not line.startswith('**') and len(line) > 1 and line[1] == ' ':
            # This is a bullet point
            if current_text:
                paragraphs.append(current_text)
                current_text = ""
            paragraphs.append(line)
        elif line.startswith('-') and len(line) > 1 and line[1] == ' ':
            # This is also a bullet point
            if current_text:
                paragraphs.append(current_text)
                current_text = ""
            paragraphs.append(line)
        elif line.startswith('#'):
            # This is a header
            if current_text:
                paragraphs.append(current_text)
                current_text = ""
            paragraphs.append(line)
        else:
            if current_text:
                current_text += " " + line
            else:
                current_text = line

    # Add any remaining text
    if current_text:
        paragraphs.append(current_text)

    return paragraphs

def create_cover_page(user_details):
    """Create a PDF cover page with user details."""
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=letter)
    width, height = letter  # Get the width and height of the page

    # Setting font
    c.setFont("Helvetica", 12)

    # Calculate the total height of all the text (20 units between each line)
    total_text_height = len(user_details) * 20

    # Starting Y position: Center the text vertically
    y = (height - total_text_height) / 2  # This centers the block of text vertically

    # Iterate over user details and calculate the center position for each line
    for key, value in user_details.items():
        text = f"{key}: {value}"

        # Calculate the width of the text to center it
        text_width = c.stringWidth(text, "Helvetica", 12)

        # Calculate X position to center the text
        x = (width - text_width) / 2

        if key == "Name":
            c.setFont("Helvetica-Bold", 14)  # Name in bold and larger font
        else:
            c.setFont("Helvetica", 12)

        c.drawString(x, y, text)
        y -= 20  # Move the next line of text lower

    c.save()
    packet.seek(0)
    return packet

def overlay_pdf(original_pdf_filename, overlay_pdf_stream, output_pdf_stream):
    """Overlay the text on top of the original PDF."""
    reader = PdfReader(original_pdf_filename)
    writer = PdfWriter()

    # Read the overlay (which is just one page)
    overlay_reader = PdfReader(overlay_pdf_stream)

    # Overlay text on top of the first page of the original PDF
    original_page = reader.pages[0]
    overlay_page = overlay_reader.pages[0]

    # Merge the overlay with the original first page
    original_page.merge_page(overlay_page)

    # Add the modified first page
    writer.add_page(original_page)

    # Add the rest of the pages from the original PDF without changes
    for page in reader.pages[1:]:
        writer.add_page(page)

    # Save the final merged PDF to the output stream
    writer.write(output_pdf_stream)
    output_pdf_stream.seek(0)

def create_content_pdf(title, subheadings, content_json):
    """Creates a PDF document with the case study content from JSON data."""
    packet = io.BytesIO()
    try:
        doc = SimpleDocTemplate(packet, pagesize=letter)
        styles = getSampleStyleSheet()

        # Define custom styles
        title_style = styles['Title']
        title_style.alignment = TA_CENTER
        title_style.spaceAfter = 0.5 * inch

        heading_style = styles['Heading2']
        heading_style.spaceBefore = 0.3 * inch
        heading_style.spaceAfter = 0.2 * inch

        body_style = styles['Normal']
        body_style.spaceAfter = 0.1 * inch
        body_style.alignment = TA_LEFT

        bullet_style = styles['Normal']
        bullet_style.leftIndent = 0.3 * inch
        bullet_style.spaceBefore = 0.05 * inch
        bullet_style.spaceAfter = 0.05 * inch

        subheading_style = styles['Heading3']
        subheading_style.spaceBefore = 0.2 * inch
        subheading_style.spaceAfter = 0.1 * inch

        story = []

        # Add Title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.2 * inch))

        # Add content for each subheading
        for subhead in subheadings:
            if subhead in content_json:
                story.append(Paragraph(subhead, heading_style))

                content = content_json[subhead]

                # Process paragraphs and handle markdown formatting
                paragraphs = process_markdown_text(content)

                for para in paragraphs:
                    para = para.strip()
                    if para.startswith('*') and not para.startswith('**') and len(para) > 1 and para[1] == ' ':
                        # Handle bullet points
                        bullet_text = para[2:].strip()  # Remove the * and the space
                        bullet_text = convert_markdown_to_html(bullet_text)  # Convert remaining markdown to HTML
                        story.append(Paragraph(f"• {bullet_text}", bullet_style))
                    elif para.startswith('-') and len(para) > 1 and para[1] == ' ':
                        # Handle bullet points with dash
                        bullet_text = para[2:].strip()  # Remove the - and the space
                        bullet_text = convert_markdown_to_html(bullet_text)  # Convert remaining markdown to HTML
                        story.append(Paragraph(f"• {bullet_text}", bullet_style))
                    elif para.startswith('#'):
                        # Handle subheadings within sections
                        header_parts = para.split(' ', 1)
                        if len(header_parts) > 1:
                            level = len(header_parts[0])  # Count the number of '#'
                            header_text = header_parts[1]
                            header_text = convert_markdown_to_html(header_text)
                            if level <= 3:
                                story.append(Paragraph(header_text, styles[f'Heading{level+1}']))
                            else:
                                story.append(Paragraph(header_text, styles['Heading4']))
                    else:
                        # Regular paragraph - apply markdown formatting
                        formatted_para = convert_markdown_to_html(para)

                        # Special case for lines that start with **bold text**
                        bold_section_match = re.match(r'^<b>(.*?)</b>:(.*)$', formatted_para)
                        if bold_section_match:
                            # This is likely a subheading with content
                            bold_text = bold_section_match.group(1)
                            content_text = bold_section_match.group(2).strip()

                            # Add the bold text as a subheading
                            story.append(Paragraph(bold_text, subheading_style))

                            # Add the content if there is any
                            if content_text:
                                story.append(Paragraph(content_text, body_style))
                        else:
                            story.append(Paragraph(formatted_para, body_style))

                story.append(Spacer(1, 0.1 * inch))

        # Build PDF
        doc.build(story)
        packet.seek(0)
        return packet
    except Exception as e:
        raise Exception(f"Error creating content PDF: {e}")

def merge_pdfs(cover_page_stream, content_pdf_stream):
    """Merge the cover page with the content PDF."""
    try:
        # Create PDF writer object
        output_pdf_stream = io.BytesIO()
        pdf_writer = PdfWriter()

        # Add cover page with user details
        cover_reader = PdfReader(cover_page_stream)
        for page in cover_reader.pages:
            pdf_writer.add_page(page)

        # Add content pages
        content_reader = PdfReader(content_pdf_stream)
        for page in content_reader.pages:
            pdf_writer.add_page(page)

        # Write the merged PDF to the output stream
        pdf_writer.write(output_pdf_stream)
        output_pdf_stream.seek(0)
        return output_pdf_stream
    except Exception as e:
        raise Exception(f"Error merging PDFs: {e}")

async def generate_case_study(job_id: str, request: CaseStudyRequest):
    """Background task to generate the case study PDF."""
    try:
        # Update job status
        jobs_status[job_id]["status"] = "processing"

        # Configure Gemini model
        model = configure_genai()

        # Prepare user details
        user_details_dict = request.user_details.dict()
        if not user_details_dict.get("date_of_submission"):
            user_details_dict["date_of_submission"] = datetime.now().strftime("%d-%m-%Y")

        # Generate content
        content_json = generate_case_study_content(
            model,
            request.title,
            request.subheadings,
            request.num_pages
        )

        if content_json is None:
            raise Exception("Failed to generate content")

        # Create cover page
        cover_page_stream = create_cover_page(user_details_dict)
        output_file_front_stream = io.BytesIO()
        overlay_pdf(FIRSTPAGE_PDF, cover_page_stream, output_file_front_stream)

        # Create content PDF
        temp_pdf_stream = create_content_pdf(request.title, request.subheadings, content_json)

        # Merge PDFs
        final_pdf_stream = merge_pdfs(output_file_front_stream, temp_pdf_stream)

        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in request.title).rstrip()
        safe_title = safe_title.replace(' ', '_')
        if not safe_title:
            safe_title = "case_study"  # fallback filename

        pdf_filename = f"{safe_title[:50]}_case_study.pdf"

        # Store the final PDF stream in job status
        jobs_status[job_id]["file_stream"] = final_pdf_stream
        jobs_status[job_id]["file_name"] = pdf_filename

        # Update job status
        jobs_status[job_id]["status"] = "completed"

    except Exception as e:
        # Update job status on error
        jobs_status[job_id]["status"] = "failed"
        jobs_status[job_id]["error"] = str(e)

# --- API Endpoints ---

@app.post("/generate", response_model=JobStatus)
async def generate_case_study_api(background_tasks: BackgroundTasks, request: CaseStudyRequest):
    """
    Submit a case study generation request.

    Returns a job ID that can be used to check status and download the PDF.
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Initialize job status
    jobs_status[job_id] = {
        "job_id": job_id,
        "status": "queued"
    }

    # Start background task
    background_tasks.add_task(generate_case_study, job_id, request)

    return jobs_status[job_id]

@app.get("/status/{job_id}", response_model=JobStatus)
async def check_status(job_id: str):
    """
    Check the status of a case study generation job.
    """
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")

    return jobs_status[job_id]

@app.get("/download/{job_id}")
async def download_pdf(job_id: str):
    """
    Download the generated PDF for a completed job.
    """
    if job_id not in jobs_status:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_status[job_id]

    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed. Current status: {job['status']}")

    if "file_stream" not in job:
        raise HTTPException(status_code=500, detail="File stream not found in job data")

    file_stream = job["file_stream"]
    file_name = job["file_name"]

    return StreamingResponse(
        file_stream,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={file_name}"}
    )
