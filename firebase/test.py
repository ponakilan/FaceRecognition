from google.cloud import firestore

db = firestore.Client()
docs = db.collection('test').where('uploaded_at', '==', 'asfasjsfajf').get()

for doc in docs:
    uploaded_time = doc.get('uploaded_at')
    print(uploaded_time)
