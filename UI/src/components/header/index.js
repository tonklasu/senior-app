import Link from 'next/link';
import Image from 'next/image';


export const Header = () => {
  return (
    <header>
      <div className='TopNav'>
        <Link href={'/'}><Image alt="logo" src={'/logo.png'} width={405/1.5} height={150/1.5} /></Link>
        <Link href={'/'}><h1>Sentiment Analysis Project</h1></Link>
        <nav >
            <ul>
              <h3>
                <Link href="/"> Home </Link> |
                <Link href="/inference-real"> Inference </Link> |
                <Link href="/about-us"> About Us </Link>
              </h3>
            </ul>
        </nav>
      </div>
    </header>
  );
};