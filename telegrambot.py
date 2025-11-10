from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, CallbackContext, filters
)
from io import BytesIO
from PIL import Image
from Models import segmentation
import cv2

# Token
TOKEN = "8351984678:AAFBWbtN8wmZpVBEdRPmKOQ7MbWdc-bL8_c"

# Predefined Data Dictionary with updated bright colors
data = {
    1: {
        "class_names": ['bg', 'Brain Tumor'],
        "weights_name": "brain_tumor.pt",
        "color_sample": [(0, 0, 0), (255, 0, 0)]
    },
    2: {
        "class_names": ["bg", "Roads"],
        "weights_name": "road.pt",
        "color_sample": [(0, 0, 0), (0, 255, 255)]
    },
    3: {
        "class_names": ['bg', 'Cracks'],
        "weights_name": "cracks.pt",
        "color_sample": [(0, 0, 0), (255, 165, 0)]
    },
    4: {     
        "class_names": ['bg', 'Leaf Disease'],
        "weights_name": "leaf_disease.pt",
        "color_sample": [(0, 0, 0), (255, 0, 0)]
    },
    5: {
        "class_names": ['bg', 'Person'],
        "weights_name": "person.pt",
        "color_sample": [(0, 0, 0), (255, 105, 180)]
    },
    6: {
        "class_names": ['bg', 'Pothole'],
        "weights_name": "pothole.pt",
        "color_sample": [(0, 0, 0), (0, 191, 255)]
    }
}


# Segmentation processor
def apply_segmentation(image, class_colors, weights_name):

    result_img = segmentation(image, class_colors, weights_name)
    result_img = cv2.cvtColor(result_img,cv2.COLOR_BGR2RGB)

    # Convert result to BytesIO 
    _, img_encoded = cv2.imencode('.jpg', result_img) 

    return BytesIO(img_encoded.tobytes())


# /start command
async def start(update: Update, context: CallbackContext):
    buttons = []

    for i in data:
        model_classes = ', '.join(data[i]['class_names']) 
        button = [InlineKeyboardButton(f"Model {i}: {model_classes}", callback_data=str(i))]
        buttons.append(button)

    reply_markup = InlineKeyboardMarkup(buttons)

    await update.message.reply_text(
        "🧠 Please select the segmentation model you want to use:",
        reply_markup=reply_markup
    )


# Button handler for model selection
async def button_handler(update: Update, context: CallbackContext):
    query = update.callback_query

    print("\n\n",query,"\n\n",query.data)

    model_id = int(query.data)

    context.user_data["selected_model"] = model_id

    model_name = data[model_id]["weights_name"]

    await query.edit_message_text(
        f"✅ Model selected: *{model_name}*\n\n📸 Now send an image for segmentation.",
        parse_mode="Markdown"
    )


# Handle uploaded photo
async def handle_photo(update: Update, context: CallbackContext):
    if "selected_model" not in context.user_data:
        await update.message.reply_text("⚠️ Please select a model first using /start before sending an image.")
        return

    model_id = context.user_data["selected_model"]
    model_data = data[model_id]

    file = await update.message.photo[-1].get_file()
    image_bytes = BytesIO(await file.download_as_bytearray())
    image = Image.open(image_bytes).convert("RGB")

    segmented_image = apply_segmentation(
        image,
        model_data["color_sample"],
        model_data["weights_name"]
    )

    await update.message.reply_photo(
        photo=segmented_image,
        caption=f"🧪 *Model Used:* `{model_data['weights_name']}`",
        parse_mode="Markdown"
    )

# # /help command
async def help_command(update: Update, context: CallbackContext):
    await update.message.reply_text("Use /start to select a model before uploading images.")


# # Main function

def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()


if __name__ == "__main__":
    main()
