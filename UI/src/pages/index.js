import Head from 'next/head'
import Image from 'next/image'
import Link from 'next/link'

function Page(){
        return(
            <>
                <Head>
                    <title>Home</title>
                </Head>
                <div className='card'>
                    <Image alt="logo" src={'/home1.png'} width={600} height={600} priority={true}/>
                    <div className='caption'>
                        <h3>Sentiment Analysis of face expression and speech signal using Machine Learning Algorithms</h3><br/>
                        <p className='prag'>This project is about Sentiment Analysis. Our team focuses is create Deep Learning model
                        that analyze emotion from facial expressions and speech signals and demonstrate the result of the input video.
                        What is the emotion of this person ? what time does this person get angry ? how long is this person be happy ?<br/><br/></p>
                        <Link href='/inference-real' className='try-button'>Try our model</Link>
                    </div>
                </div>
            </>
        )        
}

export default Page
