import os
import datetime
import logging
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

logger = logging.getLogger("ForensicANPR.Report")

class ForensicReportGenerator:
    """Enterprise-grade PDF Generator for ANPR Forensic Evidence Documentation."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._init_custom_styles()

    def _init_custom_styles(self):
        """Initializes a dark-navy corporate design system for the document."""
        self.styles.add(ParagraphStyle(
            name='ForensicHeader',
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0F172A'),
            alignment=1  # Centered
        ))
        self.styles.add(ParagraphStyle(
            name='ForensicSubheader',
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#475569'),
            alignment=1
        ))
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=10,
            spaceAfter=6
        ))
        self.styles.add(ParagraphStyle(
            name='DetailsLabel',
            fontName='Helvetica-Bold',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#334155')
        ))
        self.styles.add(ParagraphStyle(
            name='DetailsValue',
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=colors.HexColor('#0F172A')
        ))
        self.styles.add(ParagraphStyle(
            name='TamperedAlert',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#991B1B')
        ))
        self.styles.add(ParagraphStyle(
            name='CleanAlert',
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=13,
            textColor=colors.HexColor('#166534')
        ))

    def generate(self, 
                 case_id: str, 
                 investigator: str, 
                 plate_text: str, 
                 original_img_path: str = None, 
                 ela_img_path: str = None, 
                 metadata: dict = None, 
                 forensics: dict = None) -> str:
        """Compiles evidence and builds the PDF document."""
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        if metadata is None:
            metadata = {}
        if forensics is None:
            forensics = {}

        story = []

        # Title Block
        story.append(Paragraph("FORENSIC ANPR INVESTIGATION REPORT", self.styles['ForensicHeader']))
        story.append(Paragraph("CYBER CRIMES & VEHICLE INTEGRITY DIVISION", self.styles['ForensicSubheader']))
        story.append(Spacer(1, 15))

        # Case Information Table
        info_data = [
            [
                Paragraph("Case Identifier:", self.styles['DetailsLabel']),
                Paragraph(case_id, self.styles['DetailsValue']),
                Paragraph("Report Date:", self.styles['DetailsLabel']),
                Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), self.styles['DetailsValue'])
            ],
            [
                Paragraph("Lead Investigator:", self.styles['DetailsLabel']),
                Paragraph(investigator, self.styles['DetailsValue']),
                Paragraph("Detected Plate:", self.styles['DetailsLabel']),
                Paragraph(plate_text, self.styles['DetailsValue'])
            ]
        ]
        info_table = Table(info_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 2*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
            ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 15))

        # Primary Evidence Details
        story.append(Paragraph("License Plate Integrity Details", self.styles['SectionHeading']))
        
        region = metadata.get("region", "UNKNOWN")
        plate_type = metadata.get("plate_type", "Standard Private")
        watchlist_status = metadata.get("watchlist_status", "CLEAN")
        watchlist_color = self.styles['CleanAlert'] if watchlist_status == "CLEAN" else self.styles['TamperedAlert']

        evidence_data = [
            [Paragraph("Region of Origin:", self.styles['DetailsLabel']), Paragraph(region, self.styles['DetailsValue'])],
            [Paragraph("Plate Format:", self.styles['DetailsLabel']), Paragraph(plate_type, self.styles['DetailsValue'])],
            [Paragraph("Watchlist Status:", self.styles['DetailsLabel']), Paragraph(watchlist_status, watchlist_color)],
            [Paragraph("OCR Confidence:", self.styles['DetailsLabel']), Paragraph(f"{metadata.get('ocr_confidence', 0.0):.2%}", self.styles['DetailsValue'])],
        ]
        
        evidence_table = Table(evidence_data, colWidths=[2*inch, 5*inch])
        evidence_table.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(evidence_table)
        story.append(Spacer(1, 15))

        # Forensic Forgery Analysis
        story.append(Paragraph("Image Tampering & Forgery Analysis", self.styles['SectionHeading']))
        
        ela_score = forensics.get("ela_score", 0.0)
        double_jpeg = forensics.get("double_jpeg_detected", False)
        exif_tampered = forensics.get("exif_tampered", False)
        copy_move = forensics.get("copy_move_detected", False)

        def flag_text(condition):
            return Paragraph("SUSPICIOUS / MANIPULATED", self.styles['TamperedAlert']) if condition else Paragraph("PASS / AUTHENTIC", self.styles['CleanAlert'])

        forgery_data = [
            [Paragraph("Error Level Analysis (ELA):", self.styles['DetailsLabel']), flag_text(ela_score > 15.0)],
            [Paragraph("Double JPEG Compression Check:", self.styles['DetailsLabel']), flag_text(double_jpeg)],
            [Paragraph("EXIF Metadata Tampering Check:", self.styles['DetailsLabel']), flag_text(exif_tampered)],
            [Paragraph("Copy-Move Forgery Keypoint Match:", self.styles['DetailsLabel']), flag_text(copy_move)],
        ]
        
        forgery_table = Table(forgery_data, colWidths=[2.5*inch, 4.5*inch])
        forgery_table.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#F1F5F9')),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(forgery_table)
        story.append(Spacer(1, 15))

        # Embed Images side-by-side if available
        image_elements = []
        if original_img_path and os.path.exists(original_img_path):
            try:
                img1 = Image(original_img_path, width=3.2*inch, height=2.4*inch)
                image_elements.append(img1)
            except Exception as e:
                logger.error(f"Could not load original image into PDF: {e}")
        
        if ela_img_path and os.path.exists(ela_img_path):
            try:
                img2 = Image(ela_img_path, width=3.2*inch, height=2.4*inch)
                image_elements.append(img2)
            except Exception as e:
                logger.error(f"Could not load ELA image into PDF: {e}")

        if len(image_elements) > 0:
            story.append(Paragraph("Visual Evidence Artifacts", self.styles['SectionHeading']))
            
            # Place images in table
            img_table_data = [image_elements]
            img_table = Table(img_table_data, colWidths=[3.5*inch, 3.5*inch])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 15))

        # Sign-off Panel
        sig_data = [
            [Paragraph("Certified Signature:", self.styles['DetailsLabel']), Paragraph("___________________________________", self.styles['DetailsValue'])],
            [Paragraph("Forensic Expert:", self.styles['DetailsLabel']), Paragraph(f"{investigator} (Digital Forensics Division)", self.styles['DetailsValue'])]
        ]
        sig_table = Table(sig_data, colWidths=[2*inch, 5*inch])
        sig_table.setStyle(TableStyle([
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ]))
        story.append(KeepTogether([
            Spacer(1, 20),
            sig_table
        ]))

        # Build Document
        doc.build(story)
        logger.info(f"Forensic Report generated successfully at {self.output_path}")
        return self.output_path
