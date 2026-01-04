# Adapted from https://github.com/clemneo/llava-interp/blob/main/src/lvlm_lens.py

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import torch
from PIL import Image

from vlm_spectra.analysis.logit_lens import compute_logit_lens


def create_logit_lens(
    hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    norm: torch.nn.Module,
    lm_head: torch.nn.Module,
    tokenizer,
    image: Image.Image,
    token_labels: List[str],
    image_size: Tuple[int, int],
    grid_size: Tuple[int, int],
    patch_size: int,
    model_name: str,
    image_filename: str,
    prompt: str,
    save_folder: str = ".",
    misc_text: str = "",
) -> None:
    """Create an interactive logit lens visualization for any VLM."""

    all_top_tokens = compute_logit_lens(hidden_states, norm, lm_head, tokenizer, top_k=5)

    sequence_length = len(all_top_tokens[0]) if all_top_tokens else 0
    if len(token_labels) != sequence_length:
        if len(token_labels) < sequence_length:
            token_labels.extend(["<UNK>"] * (sequence_length - len(token_labels)))
        else:
            token_labels = token_labels[:sequence_length]

    image_width, image_height = image_size
    image_resized = image.resize((image_width, image_height), Image.LANCZOS)

    buffered = BytesIO()
    image_resized.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    grid_h, grid_w = grid_size
    metadata_text = f"Grid: {grid_h}×{grid_w} patches, Patch size: {patch_size}×{patch_size}px"
    if misc_text:
        metadata_text += f"<br>{misc_text}"

    html_content = _generate_html(
        img_str=img_str,
        data=all_top_tokens,
        token_labels=token_labels,
        image_width=image_width,
        image_height=image_height,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
        prompt=prompt,
        metadata_text=metadata_text,
    )

    output_filename = f"{model_name}_{Path(image_filename).stem}_logit_lens.html"
    output_path = Path(save_folder) / output_filename

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Interactive logit lens HTML has been saved to: {output_path}")


def _generate_html(
    img_str: str,
    data: List,
    token_labels: List[str],
    image_width: int,
    image_height: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    prompt: str,
    metadata_text: str,
) -> str:
    """Generate the HTML content for logit lens visualization."""

    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Logit Lens</title>
    <style>
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        .container { display: flex; }
        .image-container { 
            flex: 0 0 auto; 
            margin: 20px; 
            position: relative;
            width: ${image_width}px;
        }
        .highlight-box {
            position: absolute;
            border: 2px solid red;
            pointer-events: none;
            display: none;
        }
        .table-container { 
            flex: 1 1 auto;
            position: relative;
            max-height: 90vh;
            overflow: auto;
            margin: 20px;
        }
        table { 
            border-collapse: separate;
            border-spacing: 0;
        }
        th, td { 
            border: 1px solid #ddd; 
            padding: 8px; 
            text-align: center;
            min-width: 80px;
        }
        th { 
            background-color: #f2f2f2; 
            font-weight: bold;
        }
        .corner-header {
            position: sticky;
            top: 0;
            left: 0;
            z-index: 3;
            background-color: #f2f2f2;
        }
        .row-header {
            position: sticky;
            left: 0;
            z-index: 2;
            background-color: #f2f2f2;
        }
        .col-header {
            position: sticky;
            top: 0;
            z-index: 1;
            background-color: #f2f2f2;
        }
        #tooltip {
            display: none;
            position: fixed;
            background: white;
            border: 1px solid black;
            padding: 5px;
            z-index: 1000;
            pointer-events: none;
            max-width: 300px;
            font-size: 14px;
        }
        .highlighted-row {
            background-color: #ffff99;
        }
        .image-info {
            margin-top: 10px;
            font-size: 14px;
            width: 100%;
            word-wrap: break-word;
        }
        .prompt {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .instructions {
            font-style: italic;
        }
        .metadata {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="image-container">
            <img src="data:image/png;base64,${img_str}" alt="Input Image" style="width: ${image_width}px; height: ${image_height}px;">
            <div class="highlight-box"></div>
            <div class="image-info">
                <p class="prompt">Prompt: "${prompt}"</p>
                <p class="instructions">Instructions: Click on image to lock the patch, click on image/table to unlock</p>
                <p class="metadata">${metadata_text}</p>
            </div>
        </div>
        <div class="table-container">
            <table id="logitLens"></table>
        </div>
    </div>
    <div id="tooltip"></div>
<script>
    const data = ${json.dumps(data)};
    const tokenLabels = ${json.dumps(token_labels)};
    const tooltip = document.getElementById('tooltip');
    const highlightBox = document.querySelector('.highlight-box');
    const image = document.querySelector('.image-container img');
    const table = document.getElementById('logitLens');
    
    const imageWidth = ${image_width};
    const imageHeight = ${image_height};
    const gridH = ${grid_h};
    const gridW = ${grid_w};
    const patchSize = ${patch_size};
    
    let isLocked = false;
    let highlightedRow = null;
    let lockedPatchIndex = null;
    
    // Create table
    const cornerHeader = table.createTHead().insertRow();
    cornerHeader.insertCell().textContent = 'Token/Layer';
    cornerHeader.cells[0].classList.add('corner-header');
    
    // Create layer headers
    for (let i = 0; i < data.length; i++) {
        const th = document.createElement('th');
        th.textContent = `Layer ${i + 1}`;
        th.classList.add('col-header');
        cornerHeader.appendChild(th);
    }
    
    // Create rows with token labels
    for (let pos = 0; pos < tokenLabels.length; pos++) {
        const row = table.insertRow();
        const rowHeader = row.insertCell();
        rowHeader.textContent = tokenLabels[pos];
        rowHeader.classList.add('row-header');
        
        for (let layer = 0; layer < data.length; layer++) {
            const cell = row.insertCell();
            const topToken = data[layer][pos][0][0];
            cell.textContent = topToken;
            
            cell.addEventListener('mouseover', (e) => {
                if (!isLocked) {
                    showTooltip(e, layer, pos, false);
                }
            });
            cell.addEventListener('mousemove', updateTooltipPosition);
            cell.addEventListener('mouseout', () => {
                if (!isLocked) {
                    hideTooltip();
                }
            });
        }
    }

    function showTooltip(e, layer, pos, shouldScroll = false) {
        tooltip.innerHTML = data[layer][pos].map(([token, prob]) => `${token}: ${prob}`).join('<br>');
        tooltip.style.display = 'block';
        updateTooltipPosition(e);
        
        if (tokenLabels[pos].startsWith('<IMG')) {
            // Extract the number from token format like <IMG001> or <IMG1000>
            const match = tokenLabels[pos].match(/<IMG(\d+)>/);
            if (match) {
                const patchIndex = parseInt(match[1]);
                highlightImagePatch(patchIndex);
                highlightTableRow(pos, shouldScroll);
            }
        } else {
            highlightBox.style.display = 'none';
            unhighlightTableRow();
        }
    }

    function hideTooltip() {
        tooltip.style.display = 'none';
        if (!isLocked) {
            highlightBox.style.display = 'none';
            unhighlightTableRow();
        }
    }

    function updateTooltipPosition(e) {
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let x = e.clientX + 10;
        let y = e.clientY + 10;

        if (x + tooltipRect.width > viewportWidth) {
            x = e.clientX - tooltipRect.width - 10;
        }

        if (y + tooltipRect.height > viewportHeight) {
            y = e.clientY - tooltipRect.height - 10;
        }

        x = Math.max(0, x);
        y = Math.max(0, y);

        tooltip.style.left = x + 'px';
        tooltip.style.top = y + 'px';
    }
    
    function highlightImagePatch(patchIndex) {
        const scaleFactor = image.width / imageWidth;
        const row = Math.floor((patchIndex - 1) / gridW);
        const col = (patchIndex - 1) % gridW;
        
        const left = col * patchSize * scaleFactor;
        const top = row * patchSize * scaleFactor;
        const size = patchSize * scaleFactor;
        
        highlightBox.style.left = `${left}px`;
        highlightBox.style.top = `${top}px`;
        highlightBox.style.width = `${size}px`;
        highlightBox.style.height = `${size}px`;
        highlightBox.style.display = 'block';
    }

    function highlightTableRow(rowIndex, shouldScroll = false) {
        if (highlightedRow) {
            highlightedRow.classList.remove('highlighted-row');
        }
        highlightedRow = table.rows[rowIndex + 1];  // +1 to account for header row
        highlightedRow.classList.add('highlighted-row');
        if (shouldScroll) {
            highlightedRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    function scrollTableToRight() {
        const tableContainer = document.querySelector('.table-container');
        tableContainer.scrollLeft = tableContainer.scrollWidth;
    }

    function unhighlightTableRow() {
        if (highlightedRow) {
            highlightedRow.classList.remove('highlighted-row');
            highlightedRow = null;
        }
    }

    image.addEventListener('mousemove', (e) => {
        if (!isLocked) {
            const patchIndex = getPatchIndexFromMouseEvent(e);
            if (patchIndex > 0 && patchIndex <= gridH * gridW) {
                highlightImagePatch(patchIndex);
                const tokenIndex = getTokenIndexFromPatchIndex(patchIndex);
                if (tokenIndex !== -1) {
                    showTooltip(e, 0, tokenIndex, true);
                    scrollTableToRight();
                }
            }
        }
    });

    image.addEventListener('mouseout', () => {
        if (!isLocked) {
            hideTooltip();
        }
    });

    image.addEventListener('click', (e) => {
        isLocked = !isLocked;
        if (isLocked) {
            lockedPatchIndex = getPatchIndexFromMouseEvent(e);
            if (lockedPatchIndex > 0 && lockedPatchIndex <= gridH * gridW) {
                highlightImagePatch(lockedPatchIndex);
                const tokenIndex = getTokenIndexFromPatchIndex(lockedPatchIndex);
                if (tokenIndex !== -1) {
                    highlightTableRow(tokenIndex, true);
                }
            }
        } else {
            lockedPatchIndex = null;
            hideTooltip();
        }
    });

    table.addEventListener('click', () => {
        isLocked = false;
        lockedPatchIndex = null;
        hideTooltip();
    });

    function getPatchIndexFromMouseEvent(e) {
        const rect = image.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const scaleFactor = imageWidth / image.width;
        const patchX = Math.floor(x * scaleFactor / patchSize);
        const patchY = Math.floor(y * scaleFactor / patchSize);
        return patchY * gridW + patchX + 1;
    }

    function getTokenIndexFromPatchIndex(patchIndex) {
        // Create the expected token label format matching Python's f"<IMG{(img_token_counter+1):03d}>"
        // Python's :03d pads with zeros to minimum 3 digits, but doesn't truncate longer numbers
        const paddedIndex = patchIndex.toString().padStart(3, '0');
        const expectedLabel = `<IMG${paddedIndex}>`;
        return tokenLabels.findIndex(label => label === expectedLabel);
    }
</script>
</body>
</html>
    """

    html_content = html_template.replace("${img_str}", img_str)
    html_content = html_content.replace("${json.dumps(data)}", json.dumps(data))
    html_content = html_content.replace(
        "${json.dumps(token_labels)}", json.dumps(token_labels)
    )
    html_content = html_content.replace("${image_width}", str(image_width))
    html_content = html_content.replace("${image_height}", str(image_height))
    html_content = html_content.replace("${grid_h}", str(grid_h))
    html_content = html_content.replace("${grid_w}", str(grid_w))
    html_content = html_content.replace("${patch_size}", str(patch_size))
    html_content = html_content.replace("${prompt}", prompt.replace('"', "\\\""))
    html_content = html_content.replace("${metadata_text}", metadata_text)

    return html_content
