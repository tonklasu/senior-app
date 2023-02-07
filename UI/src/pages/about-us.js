import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';

function Page(){
    return (
        <>
            <Head>
            <title>About Us</title>
            </Head>
            <div className='card'>
                    <Image alt="o" src={'/aboutus1.gif'} width={650} height={350} priority={true}/>
                    <div className='caption'>
                        <h3>What is Sentiment Analysis ?</h3><br/>
                        <p className='prag'>Sentiment analysis using facial expressions and speech is a type of emotion recognition technology that combines two modalities - facial expressions and speech - to determine the emotional state of an individual. The goal of this technology is to provide a more complete picture of an individual's emotional state by considering both their facial expressions and speech patterns. Sentiment analysis using facial expressions and speech has potential applications in customer service, marketing, and helping individuals better understand and manage their own emotions.</p>
                    </div>
            </div>
            <div className='card'>
                    <div className='caption'>
                        <h3>Our Approach (Facial expressions)</h3><br/>
                        <p className='prag'>In the facial expressions part, we use a dataset name AffectNet-HQ and 
                        for preprocessing data we use a python library name MediaPipe face-mesh
                        to crop the area of the face and do a face alignment. We use a pre-trained model named InceptionV3
                        to extract features and add more classification layers.</p>
                    </div>
                    <Image alt="o" src={'/approach1.png'} width={620} height={320} priority={true}/>
            </div>
            <div className='card'>
                    <Image alt="o" src={'/approach2.png'} width={720} height={480} priority={true}/>
                    <div className='caption'>
                        <h3>Our Approach (Speech signals)</h3><br/>
                        <p className='prag'>In the speech part, we use a dataset name Thai Speech Emotion Dataset. 
                        We use 3 features to train the model MFCC, Mel spectrogram, and Chromagarm. 
                        A model that we use to train is CNN model.</p>
                    </div>
            </div>
            <div className='end-abuotus'>
                <h3>Conclusion</h3>
                <Image alt="o" src={'/conclusion.png'} width={1200} height={640} priority={true}/>
                <p className='prag'>The last one is to assemble the two models. We use results from the facial model and the speech model. to weigh two results to get the last result. In our project, we weight 50:50 equally.</p>
            </div>
        </>
        
    )
}
export default Page;