import axios from 'axios';
import Head from 'next/head';
import Image from 'next/image';
import { useState } from 'react';

const api_path = "139.162.10.207:8000"

function App(){
  const [imageSrc,setimageSrc] = useState(null);
  const [SelectedFile,setSelectedFile] = useState(null);
  const [UploadPercentage,setUploadPercentage] = useState(0);

  function handleSubmit(event){
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", SelectedFile);
    axios.post(`http://${api_path}/uploadfile/`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          setUploadPercentage(parseInt(Math.round((progressEvent.loaded * 100) / progressEvent.total)))
          setTimeout(() => setUploadPercentage(0), 1000);
        }
      })
      .then((res) => {
        alert("เสร็จละ");
        setimageSrc(res.data)
        setSelectedFile(null)
        console.log(res.data[2]);
      })
      .catch((err) => {
        console.log(err);
      });
  }
  function handleChange(event){
    setSelectedFile(event.target.files[0])
    console.log(event.target.files)
  };

  return (
    <>
      <Head>
        <title>Inference</title>
      </Head>


      {imageSrc === null ?(
      <div>
        <div className='result'>
          <div className='result-image'>
              <Image src={'/fer-line.png'} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={'/fer-bar.png'} width={500} height={400} alt = 'o' priority={true} />       
          </div>
        </div>
        <div className='result'>
          <div className='result-image'>
              <Image src={'/ser-line.png'} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={'/ser-bar.png'} width={500} height={400} alt = 'o' priority={true} />       
          </div>
        </div>
        <div className='result'>
          <div className='result-image'>
              <Image src={'/ferser-line.png'} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={'/ferser-bar.png'} width={500} height={400} alt = 'o' priority={true} />       
          </div>
        </div>
      </div>
      
      ):(
      <div>
        <div className='result'>
          <div className='result-image'>
              <Image src={`data:image/jpeg;base64,${imageSrc[0]}`} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={`data:image/jpeg;base64,${imageSrc[3]}`} width={600} height={400} alt = 'o' priority={true}/>       
          </div>
        </div>
        <div className='result'>
          <div className='result-image'>
              <Image src={`data:image/jpeg;base64,${imageSrc[1]}`} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={`data:image/jpeg;base64,${imageSrc[4]}`} width={600} height={400} alt = 'o' priority={true}/>       
          </div>
        </div>
        <div className='result'>
          <div className='result-image'>
              <Image src={`data:image/jpeg;base64,${imageSrc[2]}`} width={600} height={400} alt = 'o' priority={true}/>
              <Image src={`data:image/jpeg;base64,${imageSrc[5]}`} width={600} height={400} alt = 'o' priority={true}/>       
          </div>
        </div>
      </div>
      )}
      <div className='end-inference'>
        {'Upload your video here '}
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleChange} />
            <button type="submit">Upload</button>
            {UploadPercentage > 0 ? (
                <div className="progress">
                  <div
                    className="progress-bar progress-bar-striped bg-success"
                    role="progressbar"
                    style={{ width: `${UploadPercentage}%` }}>
                    {UploadPercentage}%
                  </div>
                </div> ) : null}
        </form>
      </div>
    </>
  )
}

export default App