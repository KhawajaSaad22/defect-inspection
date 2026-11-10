import tempfile
from pathlib import Path
import streamlit as st
from inspect_defects import Encoder, load, predict, overlay

st.set_page_config(page_title='Defect inspection',layout='wide')
st.title('Few-shot defect inspection')
st.caption('Normal-only reference model. Red regions indicate unusual appearance, not a confirmed defect.')
model=st.text_input('Saved model', 'artifacts/dino-pca.npz')
upload=st.file_uploader('Inspect an image',type=['png','jpg','jpeg'])
@st.cache_resource
def resources(path, modified):
    state=load(path)
    return state,Encoder(str(state['backend']),int(state['size']))
if upload and st.button('Inspect'):
    try:
        state,encoder=resources(model,Path(model).stat().st_mtime_ns)
        with tempfile.TemporaryDirectory() as folder:
            image=Path(folder)/'input.png'; image.write_bytes(upload.getvalue())
            score,heat=predict(image,state,encoder); target=Path(folder)/'overlay.png'
            overlay(image,heat,target,float(state.get('visual_high',state['threshold'])),float(state.get('visual_low',0)))
            st.image([str(image),str(target)],caption=['Input','Anomaly overlay'])
            st.write({'score':score,'threshold':float(state['threshold']),'review_needed':score>float(state['threshold'])})
    except Exception as exc: st.error(str(exc))
