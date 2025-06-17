"""
title: PDF Table to XLSX Converter
author: JavierJimenez/JJ
author_url: https://github.com/JavierJ002 (optional)
description: A powerful tool to extract tabular data from uploaded PDF documents and return it as a formatted XLSX file. Supports both lined (lattice) and unlined (stream) tables. Each table found will be placed on a separate sheet in the Excel file.
requirements: camelot-py[cv], pandas, xlsxwriter, tabulate
version: 0.0.2
license: MIT
"""

import os
import tempfile
from typing import Any, Callable, List, Dict
from pydantic import BaseModel, Field
import camelot
import pandas as pd
import xlsxwriter 
from tabulate import tabulate 



class EventEmitter:
    def __init__(self, event_emitter: Callable[[dict], Any] = None):
        self.event_emitter = event_emitter

    async def progress_update(self, description: str):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "in_progress",
                        "description": description,
                        "done": False,
                    },
                }
            )

    async def error_update(self, description: str):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "error",
                        "description": description,
                        "done": True,
                    },
                }
            )

    async def success_update(self, description: str):
        if self.event_emitter:
            await self.event_emitter(
                {
                    "type": "status",
                    "data": {
                        "status": "success",
                        "description": description,
                        "done": True,
                    },
                }
            )


class Tools:
    class Valves(BaseModel):
        DEFAULT_FLAVOR: str = Field(
            default="stream",
            description="Default Camelot table detection method: 'stream' (for tables without lines) or 'lattice' (for tables with lines).",
        )
        DEFAULT_PAGES: str = Field(
            default="all",
            description="Default pages to scan: 'all' or specific page numbers (e.g., '1', '1,3-5').",
        )
        OUTPUT_FILENAME: str = Field(
            default="extracted_pdf_tables.xlsx",
            description="Default filename for the output XLSX file."
        )

    class UserValves(BaseModel):
        # User-specific settings can override global valves or add new ones
        pass

    def __init__(self):
        self.valves = self.Valves()
        if camelot is None or pd is None or xlsxwriter is None:
            raise ImportError(
                "Required libraries (camelot-py[cv], pandas, xlsxwriter) are not installed. "
                "Please install them using: pip install 'camelot-py[cv]' pandas xlsxwriter. "
                "Also, ensure Ghostscript is installed and in your PATH."
            )

    async def convert_pdf_tables_to_xlsx(
        self,
        # Open WebUI inyecta los paths de los archivos adjuntos aquí
        __file_paths__: List[str],
        pages: str = None,
        flavor: str = None,
        output_filename: str = None,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: dict = {},
    ) -> Union[Dict[str, Any], str]: 
        """
        Extracts tabular data from an uploaded PDF document and converts it into an XLSX file.
        Each detected table will be saved as a separate sheet in the Excel workbook.
        This tool is designed to work with various table layouts, automatically detecting columns.

        :param __file_paths__: (Internal from Open WebUI) A list of local paths to files uploaded by the user.
                               This tool expects exactly one PDF file.
        :param pages: (Optional) Specific pages to scan for tables. Can be 'all', a single page number (e.g., '1'),
                      or a range (e.g., '1-3'), or a comma-separated list (e.g., '1,3,5-7').
                      Defaults to 'all' if not provided or invalid.
        :param flavor: (Optional) The method Camelot should use for table detection.
                       'stream' is for tables that don't have lines separating cells (like the example provided).
                       'lattice' is for tables that have lines separating cells.
                       Defaults to 'stream' if not provided or invalid.
        :param output_filename: (Optional) The desired name for the output XLSX file (e.g., "my_report_data.xlsx").
                                Defaults to "extracted_pdf_tables.xlsx" if not provided.
        :return: A dictionary pointing to the generated XLSX file, or an error message string.
        """
        emitter = EventEmitter(__event_emitter__)
        temp_xlsx_path = None

        if not __file_paths__:
            await emitter.error_update("No PDF file was provided. Please upload a PDF document.")
            return "Error: No PDF file was provided. Please upload a PDF document."

        
        pdf_file_path = next((f for f in __file_paths__ if f.lower().endswith('.pdf')), None)

        if not pdf_file_path or not os.path.exists(pdf_file_path):
            await emitter.error_update(f"No valid PDF file found among the uploaded files: {', '.join(__file_paths__)}")
            return f"Error: No valid PDF file found among the uploaded files: {', '.join(__file_paths__)}"

        
        effective_pages = pages if pages else self.valves.DEFAULT_PAGES
        effective_flavor = flavor if flavor in ["stream", "lattice"] else self.valves.DEFAULT_FLAVOR
        effective_output_filename = output_filename if output_filename and output_filename.endswith('.xlsx') else self.valves.OUTPUT_FILENAME

        try:
            await emitter.progress_update(f"Processing PDF file: {os.path.basename(pdf_file_path)}. Extracting tables using '{effective_flavor}' flavor on pages '{effective_pages}'...")

            tables = camelot.read_pdf(
                pdf_file_path,
                pages=effective_pages,
                flavor=effective_flavor,
                strip_text='\n' 
            )

            if not tables:
                await emitter.success_update(f"No tables found in PDF {os.path.basename(pdf_file_path)} on pages '{effective_pages}' with flavor '{effective_flavor}'.")
                return f"No tables found in PDF {os.path.basename(pdf_file_path)} on pages '{effective_pages}' with flavor '{effective_flavor}'."

            await emitter.progress_update(f"Found {len(tables)} table(s). Generating XLSX file...")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                temp_xlsx_path = tmp_file.name

            with pd.ExcelWriter(temp_xlsx_path, engine='xlsxwriter') as writer:
                for i, table in enumerate(tables):
                    df = table.df.copy()

                    original_cols = df.columns.tolist()
                    cleaned_cols = []
                    for col in original_cols:
                        if isinstance(col, tuple): 
                            col_name = " ".join(filter(None, [str(c).strip() for c in col]))
                        else:
                            col_name = str(col).strip()
                        cleaned_cols.append(col_name if col_name else f"Column_{len(cleaned_cols)}") # Assign generic name if empty

                    seen = {}
                    final_cols = []
                    for col in cleaned_cols:
                        if col in seen:
                            seen[col] += 1
                            final_cols.append(f"{col}_{seen[col]}")
                        else:
                            seen[col] = 1
                            final_cols.append(col)
                    df.columns = final_cols

                    
                    sheet_name = f"Table_Page_{table.page}_{i+1}"
                    if len(sheet_name) > 31:
                        sheet_name = sheet_name[:31] # Truncate if too long

                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            await emitter.success_update(f"Successfully extracted {len(tables)} table(s) and saved to {effective_output_filename}.")

            # Return the path to the temporary XLSX file. Open WebUI will handle serving it.
            return {
                "type": "file",
                "data": {
                    "filename": effective_output_filename,
                    "path": temp_xlsx_path 
                }
            }

        except Exception as e:
            error_message = f"Failed to extract tables from PDF: {e}. "
            if "Ghostscript is not installed" in str(e):
                error_message += "This tool requires Ghostscript to be installed on your system and available in your PATH. Please install Ghostscript (https://www.ghostscript.com/download.html) and try again."
            elif "OpenCV is not installed" in str(e):
                error_message += "This tool requires OpenCV (opencv-python) to be installed. Please ensure 'camelot-py[cv]' was installed correctly."
            await emitter.error_update(error_message)
            return f"Error: {error_message}"
        finally:
            pass

#Dependencies:
# pip install 'camelot-py[cv]' pandas xlsxwriter tabulate