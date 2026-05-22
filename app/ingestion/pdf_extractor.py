import fitz
import os

output_folder = "data/images"

os.makedirs(output_folder, exist_ok=True)



pdf_path = "data/pdfs/manual.pdf"

doc = fitz.open(pdf_path)



multimodal_data = []



for page_number, page in enumerate(doc):

    print(f"\nProcesando página {page_number + 1}")


    text_blocks = page.get_text("blocks")


    image_list = page.get_images(full=True)

    saved_images = []

    for img_index, img in enumerate(image_list):

        xref = img[0]

        base_image = doc.extract_image(xref)

        image_bytes = base_image["image"]

        image_ext = base_image["ext"]

        image_name = f"page_{page_number+1}_img_{img_index}.{image_ext}"

        image_path = os.path.join(
            output_folder,
            image_name
        )

        with open(image_path, "wb") as f:

            f.write(image_bytes)

        saved_images.append(image_path)

 

    page_data = {
        "page": page_number + 1,
        "text_blocks": text_blocks,
        "images": saved_images
    }

    multimodal_data.append(page_data)



print("\nTOTAL PÁGINAS PROCESADAS:")
print(len(multimodal_data))



for page in multimodal_data[:2]:

    print("\n" + "=" * 50)

    print(f"PÁGINA: {page['page']}")

    print(f"IMÁGENES: {page['images']}")

    print("\nTEXT BLOCKS:")

    for block in page["text_blocks"][:2]:

        print(block)