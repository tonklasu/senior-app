import { Footer } from '../components/footer/index';
import { Header } from '../components/header/index';

const MainLayout = ({ children }) => {
  return (
    <>
    <main>
      <Header />
      {children}
      <Footer />
    </main>
    </>
  );
};

export default MainLayout;