import time
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tflite_runtime.interpreter as tflite

max_len = 20

FEATURE_GENERATION_MODEL_TFLITE = 'feature.tflite'
CAPTION_GENERATION_MODEL_TFLITE = 'caption.tflite'

word_to_idx = np.load('weights/word_to_idx.npy', allow_pickle = True).item()
idx_to_word = np.load('weights/idx_to_word.npy', allow_pickle = True).item()


def pad_sequences(sequences, maxlen = None, dtype = 'int32', padding = 'pre', truncating = 'pre', value = 0.):

    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)


    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

def predict_caption(upload_image):
    a = ImageOps.fit(upload_image, (300, 300), Image.ANTIALIAS)
    a = a.resize((300, 300))
    a = np.asarray(a, dtype = 'float32')
    imgp = a.reshape(1, 300, 300, 3)
    
    feat_interpreter = tflite.Interpreter(model_path = FEATURE_GENERATION_MODEL_TFLITE)
    feat_interpreter.allocate_tensors()

    input_index = feat_interpreter.get_input_details()[0]['index']
    output_index = feat_interpreter.get_output_details()[0]['index']

    feat_interpreter.set_tensor(input_index, imgp)
    feat_interpreter.invoke()
    
    feature_vector = feat_interpreter.get_tensor(output_index)
    feature_vector = feature_vector.reshape((1, 1536))
    in_text = 'startseq'
    
    for i in range(max_len):
        seq = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        seq = pad_sequences([seq], maxlen = max_len, padding = 'post')
        
        cap_interpreter = tflite.Interpreter(model_path = CAPTION_GENERATION_MODEL_TFLITE)
        cap_interpreter.allocate_tensors()

        input_index1 = cap_interpreter.get_input_details()[0]['index']
        input_index2 = cap_interpreter.get_input_details()[1]['index']
        output_index = cap_interpreter.get_output_details()[0]['index']
        
        cap_interpreter.set_tensor(input_index1, feature_vector)
        cap_interpreter.set_tensor(input_index2, np.float32(seq))
        cap_interpreter.invoke()
        
        y_pred = cap_interpreter.get_tensor(output_index)
        y_pred = y_pred.argmax()
        
        word = idx_to_word[y_pred]
        in_text += ' '+word
        
        if word == 'endseq':
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption


st.header('Image Caption Generator')

image_file = st.file_uploader('', type = ["png", "jpg", "jpeg"])
st.set_option('deprecation.showfileUploaderEncoding', False)

if image_file is None:
    st.text("Please upload an image file above")

else:
    image = Image.open(image_file)
    st.image(image, use_column_width = True)

    with st.spinner('Generating Caption...'):
        predictions = predict_caption(image)
    st.success('Done!')
    st.snow()

    st.write('Caption:')
    st.write(predictions)


with st.sidebar:
    st.title('About the Flickr30K Dataset')
    st.write('The Flickr30k dataset is a popular benchmark for sentence-based picture portrayal. The dataset has over 31,000 images. Each image in the dataset has five reference sentences provided by human annotators, resulting in nearly 1,55,000 captions.')
    st.title('Model Info')

    st.markdown("""
    - BLEU Score: **2.2**
    - Embedding Used: **GLOVE 100D**
    - Image Model used: **EfficientNetV2**""")

    st.title('About Me')
    st.markdown("""
    - Adarsh Wase
    - Current Student, IIM Indore
    - Ex - Hewlett Packard Enterprise Intern""")